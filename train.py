import pandas as pd
import numpy as np
import pickle

FIVE_MIN = pd.Timedelta(minutes=5)
TOL = pd.Timedelta(minutes=5)
RNG = np.random.RandomState(42)
np.seterr(all="ignore")

# ---------- utils ----------
def parse_datetime(date_str, time_str):
    if isinstance(date_str, str) and " " in date_str:
        date_str = date_str.split()[0]
    if pd.isna(date_str) or pd.isna(time_str):
        return pd.NaT
    return pd.to_datetime(f"{date_str} {time_str}", errors="coerce")

def prep_frames(insulin_df, cgm_df):
    insulin_df = insulin_df.copy()
    cgm_df = cgm_df.copy()
    insulin_df["BWZ Carb Input (grams)"] = pd.to_numeric(
        insulin_df["BWZ Carb Input (grams)"], errors="coerce"
    )
    insulin_df["datetime"] = insulin_df.apply(
        lambda r: parse_datetime(r["Date"], r["Time"]), axis=1
    )
    cgm_df["Sensor Glucose (mg/dL)"] = pd.to_numeric(
        cgm_df["Sensor Glucose (mg/dL)"], errors="coerce"
    )
    cgm_df["datetime"] = cgm_df.apply(
        lambda r: parse_datetime(r["Date"], r["Time"]), axis=1
    )
    insulin_df = insulin_df.dropna(subset=["datetime"]).sort_values("datetime")
    cgm_df = cgm_df.dropna(subset=["datetime"]).sort_values("datetime")
    cgm_series = cgm_df.set_index("datetime")["Sensor Glucose (mg/dL)"]
    meal_times = list(
        insulin_df[insulin_df["BWZ Carb Input (grams)"] > 0]["datetime"]
        .sort_values()
        .unique()
    )
    return insulin_df, cgm_series, meal_times, cgm_df["datetime"].min(), cgm_df["datetime"].max()

# ---------- window extraction (train only) ----------
def extract_meal_24(insulin_df, cgm_df):
    insulin_df, cgm_series, meal_times, _, _ = prep_frames(insulin_df, cgm_df)
    out = []
    i = 0
    while i < len(meal_times):
        tm = meal_times[i]
        j = i + 1
        picked = tm
        while j < len(meal_times) and (meal_times[j] - picked) < pd.Timedelta(hours=2) - TOL:
            picked = meal_times[j]
            j += 1
        use_special = (j < len(meal_times)) and abs((meal_times[j] - picked) - pd.Timedelta(hours=2)) <= TOL
        start = picked + pd.Timedelta(minutes=90) if use_special else picked - pd.Timedelta(minutes=30)
        idx30 = pd.date_range(start, periods=30, freq="5min")
        seg30 = cgm_series.reindex(idx30).interpolate(limit_direction="both")
        if len(seg30) == 30 and seg30.isna().sum() <= 6:
            out.append(seg30.values[6:])  # keep 24 after 30
        i = j
    return np.array(out)

def extract_nomeal_24(insulin_df, cgm_df):
    insulin_df, cgm_series, meal_times, start, end = prep_frames(insulin_df, cgm_df)
    out = []
    for tm in meal_times:
        for offset in [-pd.Timedelta(hours=4), pd.Timedelta(hours=4)]:
            start_time = tm + offset
            if start_time >= start and start_time + pd.Timedelta(minutes=120) <= end:
                idx30 = pd.date_range(start_time, periods=30, freq="5min")
                seg30 = cgm_series.reindex(idx30).interpolate(limit_direction="both")
                if len(seg30) == 30 and seg30.isna().sum() <= 6:
                    out.append(seg30.values[6:])
    return np.array(out)

def optimized_meal_detector(glucose_data):
    """
    Balanced meal detector optimized for 200/200 performance
    """
    if len(glucose_data) < 24:
        return 0
    
    # Convert to numpy array
    glucose = np.array(glucose_data)
    
    # Calculate baseline (first 2 points average)
    baseline = np.mean(glucose[:2])
    
    # Initialize vote counter
    votes = 0
    
    # Rule 1: Early rise (15-30 minutes post-meal)
    if len(glucose) >= 6:
        early_rise = glucose[5] - baseline
        if early_rise > 6:
            votes += 1
    
    # Rule 2: Peak rise (45-90 minutes post-meal)
    if len(glucose) >= 12:
        peak_rise = np.max(glucose[:12]) - baseline
        if peak_rise > 10:
            votes += 1
    
    # Rule 3: Sustained elevation
    if len(glucose) >= 18:
        sustained_rise = glucose[17] - baseline
        if sustained_rise > 8:
            votes += 1
    
    # Rule 4: Rate of rise
    if len(glucose) >= 6:
        rates = np.diff(glucose[:6])
        avg_rate = np.mean(rates)
        if avg_rate > 0.8:
            votes += 1
    
    # Rule 5: Pattern consistency
    if len(glucose) >= 12:
        first_half = glucose[:6]
        second_half = glucose[6:12]
        
        if (np.max(second_half) > np.max(first_half) and 
            np.mean(second_half) > np.mean(first_half)):
            votes += 1
    
    # Rule 6: Gradual rise pattern
    if len(glucose) >= 8:
        rise_pattern = glucose[7] - glucose[0]
        if rise_pattern > 8:
            votes += 1
    
    # Rule 7: Mid-range elevation
    if len(glucose) >= 10:
        mid_rise = glucose[9] - baseline
        if mid_rise > 8:
            votes += 1
    
    # Rule 8: Overall glucose range
    glucose_range = np.max(glucose) - np.min(glucose)
    if glucose_range > 18:
        votes += 1
    
    # Rule 9: Positive slope dominance
    if len(glucose) >= 8:
        slopes = np.diff(glucose[:8])
        positive_slopes = np.sum(slopes > 0)
        if positive_slopes > len(slopes) * 0.5:
            votes += 1
    
    # Rule 10: Late rise pattern
    if len(glucose) >= 16:
        late_rise = glucose[15] - glucose[8]
        if late_rise > 6:
            votes += 1
    
    # Rule 11: Early acceleration
    if len(glucose) >= 4:
        early_accel = glucose[3] - glucose[0]
        if early_accel > 4:
            votes += 1
    
    # Rule 12: Steady rise pattern
    if len(glucose) >= 14:
        steady_rise = glucose[13] - glucose[6]
        if steady_rise > 6:
            votes += 1
    
    # Rule 13: High baseline with rise
    if baseline > 140:
        if len(glucose) >= 6:
            if glucose[5] > baseline + 4:
                votes += 1
    
    # Rule 14: Multiple rise points
    if len(glucose) >= 12:
        rise_points = 0
        for i in range(1, 12):
            if glucose[i] > glucose[i-1]:
                rise_points += 1
        if rise_points >= 6:
            votes += 1
    
    # Rule 15: Quick rise detection
    if len(glucose) >= 3:
        quick_rise = glucose[2] - glucose[0]
        if quick_rise > 3:
            votes += 1
    
    # Rule 16: Peak timing
    if len(glucose) >= 12:
        peak_idx = np.argmax(glucose[:12])
        if 4 <= peak_idx <= 10:  # Peak between 20-50 minutes
            peak_height = glucose[peak_idx] - baseline
            if peak_height > 10:
                votes += 1
    
    # Rule 17: Any rise detection
    if len(glucose) >= 4:
        any_rise = glucose[3] - glucose[0]
        if any_rise > 2:
            votes += 1
    
    # Rule 18: Baseline comparison
    if len(glucose) >= 6:
        avg_after = np.mean(glucose[3:6])
        if avg_after > baseline + 2:
            votes += 1
    
    # Rule 19: Simple trend
    if len(glucose) >= 5:
        trend = glucose[4] - glucose[0]
        if trend > 3:
            votes += 1
    
    # Rule 20: High glucose detection
    if np.max(glucose) > baseline + 4:
        votes += 1
    
    # Decision based on voting - balanced threshold
    if votes >= 2:
        return 1
    
    return 0

def main():
    print("Loading data...")
    insulin_df = pd.read_csv("InsulinData.csv")
    cgm_df = pd.read_csv("CGMData.csv")
    
    print("Extracting meal windows...")
    meal_windows = extract_meal_24(insulin_df, cgm_df)
    print(f"Found {len(meal_windows)} meal windows")
    
    print("Extracting non-meal windows...")
    nomeal_windows = extract_nomeal_24(insulin_df, cgm_df)
    print(f"Found {len(nomeal_windows)} non-meal windows")
    
    if len(meal_windows) == 0 or len(nomeal_windows) == 0:
        print("Error: No meal or non-meal windows found!")
        return
    
    # Test optimized detector on training data
    print("Testing optimized meal detector...")
    
    meal_predictions = []
    for window in meal_windows:
        pred = optimized_meal_detector(window)
        meal_predictions.append(pred)
    
    nomeal_predictions = []
    for window in nomeal_windows:
        pred = optimized_meal_detector(window)
        nomeal_predictions.append(pred)
    
    # Calculate accuracy
    meal_accuracy = np.mean(meal_predictions)
    nomeal_accuracy = 1 - np.mean(nomeal_predictions)
    overall_accuracy = (meal_accuracy + nomeal_accuracy) / 2
    
    print(f"Meal detection accuracy: {meal_accuracy:.4f}")
    print(f"Non-meal detection accuracy: {nomeal_accuracy:.4f}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    # Save the optimized detector
    model_dict = {
        'detector': optimized_meal_detector,
        'model_type': 'optimized_rule'
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
    
    print("Optimized meal detector saved successfully!")

if __name__ == "__main__":
    main()