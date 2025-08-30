import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

# ---------- rule-based features ----------
def features_24(M):
    feats = []
    for row in M:
        row = np.nan_to_num(row.astype(float))
        
        # Basic glucose statistics
        mean_glucose = float(np.mean(row))
        std_glucose = float(np.std(row))
        min_glucose = float(np.min(row))
        max_glucose = float(np.max(row))
        glucose_range = max_glucose - min_glucose
        
        # Slope analysis
        slopes = np.diff(row)
        max_slope = float(np.max(slopes))
        min_slope = float(np.min(slopes))
        mean_slope = float(np.mean(slopes))
        
        # Early rise detection (first 6 points)
        if len(row) >= 6:
            early_rise = (row[5] - row[0]) / 5.0
            early_rise_abs = abs(early_rise)
        else:
            early_rise = 0.0
            early_rise_abs = 0.0
        
        # Peak detection
        peaks, _ = find_peaks(row, height=mean_glucose + 0.5*std_glucose, distance=3)
        num_peaks = len(peaks)
        
        # Time to peak
        peak_idx = np.argmax(row)
        time_to_peak = float(peak_idx) / len(row)
        
        # Area under curve
        auc = float(np.trapz(row))
        
        # Glucose variability
        glucose_cv = std_glucose / mean_glucose if mean_glucose > 0 else 0.0
        
        # Rule-based features
        # 1. Strong early rise indicator
        strong_early_rise = 1.0 if early_rise > 5.0 else 0.0
        
        # 2. Sustained rise indicator
        if len(row) >= 12:
            sustained_rise = 1.0 if (row[11] - row[0]) > 10.0 else 0.0
        else:
            sustained_rise = 0.0
        
        # 3. Peak height indicator
        peak_height = 1.0 if glucose_range > 15.0 else 0.0
        
        # 4. Slope consistency
        positive_slopes = np.sum(slopes > 0)
        slope_consistency = positive_slopes / len(slopes) if len(slopes) > 0 else 0.0
        
        # 5. Glucose level indicator
        high_glucose = 1.0 if mean_glucose > 150.0 else 0.0
        
        # 6. Variability indicator
        high_variability = 1.0 if glucose_cv > 0.1 else 0.0
        
        feat = [
            mean_glucose, std_glucose, min_glucose, max_glucose, glucose_range,
            max_slope, min_slope, mean_slope, early_rise, early_rise_abs,
            num_peaks, time_to_peak, auc, glucose_cv,
            strong_early_rise, sustained_rise, peak_height, 
            slope_consistency, high_glucose, high_variability
        ]
        feats.append(feat)
    return np.array(feats, dtype=float)

def rule_based_predict(features):
    """Simple rule-based prediction as fallback"""
    predictions = []
    for feat in features:
        # Rule 1: Strong early rise
        if feat[14] > 0:  # strong_early_rise
            predictions.append(1)
        # Rule 2: Sustained rise with good peak
        elif feat[15] > 0 and feat[17] > 0:  # sustained_rise and peak_height
            predictions.append(1)
        # Rule 3: High glucose with high variability
        elif feat[18] > 0 and feat[19] > 0:  # high_glucose and high_variability
            predictions.append(1)
        # Rule 4: Good slope consistency with early rise
        elif feat[17] > 0.6 and feat[9] > 3.0:  # slope_consistency and early_rise_abs
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

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
    
    print("Extracting features...")
    meal_features = features_24(meal_windows)
    nomeal_features = features_24(nomeal_windows)
    
    print(f"Meal features shape: {meal_features.shape}")
    print(f"Non-meal features shape: {nomeal_features.shape}")
    
    # Combine features
    X = np.vstack([meal_features, nomeal_features])
    y = np.hstack([np.ones(len(meal_features)), np.zeros(len(nomeal_features))])
    
    print(f"Total samples: {len(X)}")
    print(f"Meal samples: {np.sum(y == 1)}")
    print(f"Non-meal samples: {np.sum(y == 0)}")
    
    # Test rule-based approach
    print("Testing rule-based approach...")
    rule_predictions = rule_based_predict(X)
    rule_accuracy = np.mean(y == rule_predictions)
    rule_f1 = f1_score(y, rule_predictions)
    
    print(f"Rule-based Accuracy: {rule_accuracy:.4f}")
    print(f"Rule-based F1 Score: {rule_f1:.4f}")
    
    # Train ML model as backup
    print("Training ML model...")
    ml_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    
    ml_model.fit(X, y)
    ml_predictions = ml_model.predict(X)
    ml_accuracy = np.mean(y == ml_predictions)
    ml_f1 = f1_score(y, ml_predictions)
    
    print(f"ML Accuracy: {ml_accuracy:.4f}")
    print(f"ML F1 Score: {ml_f1:.4f}")
    
    # Choose the better approach
    if rule_f1 > ml_f1:
        print("Using rule-based approach")
        model_type = 'rule'
        threshold = 0.5
    else:
        print("Using ML approach")
        model_type = 'ml'
        threshold = 0.5
    
    # Save model
    model_dict = {
        'model': ml_model,
        'model_type': model_type,
        'threshold': threshold,
        'rule_predict': rule_based_predict
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
    
    print("Model saved successfully!")
    print(f"Model type: {model_type}")
    print(f"Threshold: {threshold}")

if __name__ == "__main__":
    main()