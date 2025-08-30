import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

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

# ---------- enhanced features ----------
def features_24(M):
    feats = []
    for row in M:
        row = np.nan_to_num(row.astype(float))
        
        # Apply smoothing for better feature extraction
        try:
            row_smooth = savgol_filter(row, window_length=5, polyorder=2)
        except:
            row_smooth = row
            
        baseline = float(np.mean(row))
        x = row - baseline
        x_smooth = row_smooth - baseline

        # Basic statistical features
        mean = float(np.mean(x))
        std = float(np.std(x))
        vmin = float(np.min(x))
        vmax = float(np.max(x))
        range_val = vmax - vmin
        q1, q3 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
        iqr = q3 - q1
        skewness = float(skew(x))
        kurt = float(kurtosis(x))
        
        # Enhanced slope features
        slopes = np.diff(x)
        max_slope = float(np.max(slopes))
        min_slope = float(np.min(slopes))
        mean_slope = float(np.mean(slopes))
        std_slope = float(np.std(slopes))
        
        # Additional slope features
        slopes15 = np.diff(x[:15])  # First 15 points
        max_slope15 = float(np.max(slopes15))
        mean_slope15 = float(np.mean(slopes15))
        
        # Rate of change features
        roc = np.abs(np.diff(x))
        roc_mean = float(np.mean(roc))
        roc_std = float(np.std(roc))
        
        # Glucose variability
        glucose_cv = float(std / (abs(mean) + 1e-8))
        
        # Time-based features
        peak_idx = np.argmax(x)
        time_to_peak = float(peak_idx / len(x))
        
        # Smoothness features
        smoothness = float(np.std(np.diff(x_smooth)))
        
        # Peak detection features
        peaks, _ = find_peaks(x, height=0, distance=3)
        num_peaks = len(peaks)
        if num_peaks > 0:
            peak_heights = x[peaks]
            max_peak_height = float(np.max(peak_heights))
            mean_peak_height = float(np.mean(peak_heights))
        else:
            max_peak_height = 0.0
            mean_peak_height = 0.0
        
        # FFT features
        fft_vals = np.abs(fft(x))
        fft_mean = float(np.mean(fft_vals[:12]))  # First 12 components
        fft_std = float(np.std(fft_vals[:12]))
        
        # Additional statistical features
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        
        # Area under curve features
        auc = float(np.trapz(x))
        auc_positive = float(np.trapz(np.maximum(x, 0)))
        auc_negative = float(np.trapz(np.minimum(x, 0)))
        
        # Velocity and acceleration features
        velocity = np.diff(x)
        acceleration = np.diff(velocity)
        max_velocity = float(np.max(np.abs(velocity)))
        max_acceleration = float(np.max(np.abs(acceleration)))
        
        # Entropy-like features
        hist, _ = np.histogram(x, bins=10)
        hist = hist / np.sum(hist)
        entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
        
        feats.append([
            mean, std, vmin, vmax, range_val, q1, q3, iqr, skewness, kurt,
            max_slope, min_slope, mean_slope, std_slope, max_slope15, mean_slope15,
            roc_mean, roc_std, glucose_cv, time_to_peak, smoothness,
            num_peaks, max_peak_height, mean_peak_height,
            fft_mean, fft_std, median, mad, auc, auc_positive, auc_negative,
            max_velocity, max_acceleration, entropy
        ])
    return np.array(feats)

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
    
    # Load the trained model
    print("Loading trained model...")
    with open("model.pkl", "rb") as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
        threshold = model_dict['threshold']
    
    # Simple holdout validation (since we have very few samples)
    print("\n=== HOLDOUT VALIDATION PERFORMANCE ===")
    # Use 80% for training, 20% for testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train a new model on the training set
    model_holdout = clone(model)
    model_holdout.fit(X_train, y_train)
    
    # Test on holdout set
    y_proba_holdout = model_holdout.predict_proba(X_test)[:, 1]
    y_pred_holdout = (y_proba_holdout > threshold).astype(int)
    
    holdout_accuracy = accuracy_score(y_test, y_pred_holdout)
    holdout_f1 = f1_score(y_test, y_pred_holdout)
    
    print(f"Holdout Accuracy: {holdout_accuracy:.4f}")
    print(f"Holdout F1 Score: {holdout_f1:.4f}")
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Training meal samples: {np.sum(y_train == 1)}, Test meal samples: {np.sum(y_test == 1)}")
    
    # Final model performance on full dataset
    print("\n=== FINAL MODEL PERFORMANCE ===")
    # Get probabilities and apply threshold
    y_proba = model.predict_proba(X)[:, 1]
    y_pred_full = (y_proba > threshold).astype(int)
    final_accuracy = accuracy_score(y, y_pred_full)
    final_f1 = f1_score(y, y_pred_full)
    
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Final F1 Score: {final_f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred_full))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred_full)
    print(cm)
    
    # Calculate percentage scores (assuming 200 total test cases)
    total_test_cases = 200
    correct_predictions = int(final_accuracy * len(X))
    score_out_of_200 = int((correct_predictions / len(X)) * total_test_cases)
    
    print(f"\n=== SCORE SUMMARY ===")
    print(f"Correct predictions: {correct_predictions}/{len(X)}")
    print(f"Estimated score out of 200: {score_out_of_200}/200")
    print(f"Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"F1 Score: {final_f1:.4f}")

if __name__ == "__main__":
    main()