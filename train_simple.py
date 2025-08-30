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

def simple_meal_detector(glucose_data):
    """
    Simple rule-based meal detector based on glucose rise patterns
    """
    if len(glucose_data) < 24:
        return 0
    
    # Convert to numpy array
    glucose = np.array(glucose_data)
    
    # Rule 1: Check for significant rise in first 6 points (30 minutes)
    if len(glucose) >= 6:
        early_rise = glucose[5] - glucose[0]
        if early_rise > 20:  # Rise of more than 20 mg/dL in 30 minutes
            return 1
    
    # Rule 2: Check for sustained rise over 12 points (1 hour)
    if len(glucose) >= 12:
        sustained_rise = glucose[11] - glucose[0]
        if sustained_rise > 30:  # Rise of more than 30 mg/dL in 1 hour
            return 1
    
    # Rule 3: Check for peak and fall pattern
    if len(glucose) >= 18:
        peak_idx = np.argmax(glucose[:18])
        if peak_idx >= 6 and peak_idx <= 12:  # Peak between 30-60 minutes
            peak_height = glucose[peak_idx] - glucose[0]
            if peak_height > 25:  # Peak rise of more than 25 mg/dL
                return 1
    
    # Rule 4: Check for overall glucose range
    glucose_range = np.max(glucose) - np.min(glucose)
    if glucose_range > 35:  # Total range more than 35 mg/dL
        return 1
    
    # Rule 5: Check for slope consistency (more positive slopes than negative)
    if len(glucose) >= 6:
        slopes = np.diff(glucose[:12])  # First 12 points
        positive_slopes = np.sum(slopes > 0)
        if positive_slopes > len(slopes) * 0.7:  # More than 70% positive slopes
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
    
    # Test simple detector on training data
    print("Testing simple meal detector...")
    
    meal_predictions = []
    for window in meal_windows:
        pred = simple_meal_detector(window)
        meal_predictions.append(pred)
    
    nomeal_predictions = []
    for window in nomeal_windows:
        pred = simple_meal_detector(window)
        nomeal_predictions.append(pred)
    
    # Calculate accuracy
    meal_accuracy = np.mean(meal_predictions)
    nomeal_accuracy = 1 - np.mean(nomeal_predictions)
    overall_accuracy = (meal_accuracy + nomeal_accuracy) / 2
    
    print(f"Meal detection accuracy: {meal_accuracy:.4f}")
    print(f"Non-meal detection accuracy: {nomeal_accuracy:.4f}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    # Save the simple detector
    model_dict = {
        'detector': simple_meal_detector,
        'model_type': 'simple_rule'
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
    
    print("Simple meal detector saved successfully!")

if __name__ == "__main__":
    main()