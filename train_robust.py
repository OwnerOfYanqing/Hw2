import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone

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

# ---------- simple but robust features ----------
def features_24(M):
    feats = []
    for row in M:
        row = np.nan_to_num(row.astype(float))
        
        # Basic statistical features
        mean = float(np.mean(row))
        std = float(np.std(row))
        vmin = float(np.min(row))
        vmax = float(np.max(row))
        range_val = vmax - vmin
        
        # Slope features
        slopes = np.diff(row)
        max_slope = float(np.max(slopes))
        min_slope = float(np.min(slopes))
        mean_slope = float(np.mean(slopes))
        
        # Peak features
        peaks, _ = find_peaks(row, height=mean + 0.5*std, distance=3)
        num_peaks = len(peaks)
        
        # Time to peak
        peak_idx = np.argmax(row)
        time_to_peak = float(peak_idx) / len(row)
        
        # Area under curve
        auc = float(np.trapz(row))
        
        # Glucose rise rate (first 6 points)
        if len(row) >= 6:
            rise_rate = (row[5] - row[0]) / 5.0
        else:
            rise_rate = 0.0
            
        # Glucose fall rate (last 6 points)
        if len(row) >= 6:
            fall_rate = (row[-1] - row[-6]) / 5.0
        else:
            fall_rate = 0.0
        
        # Simple features that are robust
        feat = [
            mean, std, vmin, vmax, range_val,
            max_slope, min_slope, mean_slope,
            num_peaks, time_to_peak, auc,
            rise_rate, fall_rate
        ]
        feats.append(feat)
    return np.array(feats, dtype=float)

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
    
    # Create a simple but robust model
    print("Training robust model...")
    
    # Use a simple Random Forest with regularization
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Limit depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    # Train the model
    model.fit(X, y)
    
    # Evaluate on training data
    y_pred = model.predict(X)
    train_accuracy = np.mean(y == y_pred)
    train_f1 = f1_score(y, y_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training F1 Score: {train_f1:.4f}")
    
    # Save model with threshold
    threshold = 0.5  # Use default threshold for simplicity
    model_dict = {
        'model': model,
        'threshold': threshold
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
    
    print("Model saved successfully!")
    print(f"Model threshold: {threshold}")

if __name__ == "__main__":
    main()