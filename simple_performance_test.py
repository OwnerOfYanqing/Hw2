import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

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
        rge = vmax - vmin
        rms = float(np.sqrt(np.mean(x**2)))
        skw = float(skew(x, bias=False)) if std > 1e-12 else 0.0
        kur = float(kurtosis(x, bias=False)) if std > 1e-12 else 0.0

        # Enhanced derivative features
        dx = np.diff(x)
        dv_mean = float(np.mean(dx)) if dx.size else 0.0
        dv_std  = float(np.std(dx))  if dx.size else 0.0
        dv_max  = float(np.max(dx))  if dx.size else 0.0
        dv_min  = float(np.min(dx))  if dx.size else 0.0
        ddx = np.diff(dx)
        da_std = float(np.std(ddx)) if ddx.size else 0.0

        # Area under curve
        try:
            auc = float(np.trapezoid(x))
        except Exception:
            auc = float(np.trapz(x))

        # Peak analysis
        peaks, _ = find_peaks(x, height=0.5*std, distance=3)
        num_peaks = int(len(peaks))
        t_max = int(np.argmax(x))
        pos_sum = float(np.sum(dx[dx > 0])) if dx.size else 0.0

        # Enhanced slope features
        max_slope30 = 0.0
        max_slope15 = 0.0
        for k in range(0, len(x) - 6):
            slope6 = (x[k + 6] - x[k]) / 6.0
            max_slope30 = max(max_slope30, slope6)
        for k in range(0, len(x) - 3):
            slope3 = (x[k + 3] - x[k]) / 3.0
            max_slope15 = max(max_slope15, slope3)

        # FFT features
        fft_mag = np.abs(fft(x))
        if fft_mag.size < 7:
            fft_feats = np.pad(fft_mag[1:], (0, 6 - max(0, fft_mag.size - 1)))
        else:
            fft_feats = fft_mag[1:7]

        # Additional features
        # Rate of change features
        rate_of_change = np.diff(x_smooth)
        roc_mean = float(np.mean(rate_of_change)) if rate_of_change.size else 0.0
        roc_std = float(np.std(rate_of_change)) if rate_of_change.size else 0.0
        
        # Glucose variability
        glucose_cv = float(std / baseline) if baseline > 0 else 0.0
        
        # Time-based features
        time_to_peak = float(t_max) / len(x) if len(x) > 0 else 0.0
        
        # Smoothness features
        smoothness = float(np.std(np.diff(x_smooth))) if len(x_smooth) > 1 else 0.0
        
        # Energy features
        energy = float(np.sum(x**2))
        
        # Entropy-like features
        hist, _ = np.histogram(x, bins=10)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log(hist + 1e-10))) if hist.size > 0 else 0.0

        feat = [
            baseline, mean, std, vmin, vmax, rge, rms, skw, kur,
            dv_mean, dv_std, dv_max, dv_min, da_std,
            auc, num_peaks, t_max, pos_sum, max_slope30, max_slope15,
            float(fft_feats[0]), float(fft_feats[1]), float(fft_feats[2]),
            float(fft_feats[3]), float(fft_feats[4]), float(fft_feats[5]),
            roc_mean, roc_std, glucose_cv, time_to_peak, smoothness, energy, entropy
        ]
        feats.append(np.nan_to_num(feat))
    return np.asarray(feats, dtype=float)

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
    
    # Test model performance on available data
    print("\n=== MODEL PERFORMANCE ON AVAILABLE DATA ===")
    
    # Get probabilities and apply threshold
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # Show predictions for each sample
    print("\n=== DETAILED PREDICTIONS ===")
    for i in range(len(X)):
        sample_type = "MEAL" if y[i] == 1 else "NON-MEAL"
        pred_type = "MEAL" if y_pred[i] == 1 else "NON-MEAL"
        prob = y_proba[i]
        correct = "✓" if y[i] == y_pred[i] else "✗"
        print(f"Sample {i+1} ({sample_type}): Predicted {pred_type} (prob={prob:.3f}) {correct}")
    
    # Calculate estimated score out of 200
    total_test_cases = 200
    correct_predictions = int(accuracy * len(X))
    score_out_of_200 = int((correct_predictions / len(X)) * total_test_cases)
    
    print(f"\n=== SCORE SUMMARY ===")
    print(f"Correct predictions: {correct_predictions}/{len(X)}")
    print(f"Estimated score out of 200: {score_out_of_200}/200")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    
    # Test with the sample test data
    print("\n=== TESTING WITH SAMPLE DATA ===")
    try:
        test_data = pd.read_csv("test_sample.csv", header=None)
        print(f"Test data shape: {test_data.shape}")
        
        # Extract features from test data
        test_features = features_24(test_data.values)
        print(f"Test features shape: {test_features.shape}")
        
        # Make predictions
        test_proba = model.predict_proba(test_features)[:, 1]
        test_pred = (test_proba > threshold).astype(int)
        
        print("Test predictions:")
        for i in range(len(test_pred)):
            pred_type = "MEAL" if test_pred[i] == 1 else "NON-MEAL"
            prob = test_proba[i]
            print(f"Test sample {i+1}: Predicted {pred_type} (prob={prob:.3f})")
            
    except Exception as e:
        print(f"Could not test with sample data: {e}")

if __name__ == "__main__":
    main()