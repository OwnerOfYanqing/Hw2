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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
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

# ---------- window extraction ----------
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
            out.append(seg30.values[6:])  # keep 24 after event
        i = j
    return np.asarray(out)

def _collect_nomeal(insulin_df, cgm_series, start, max_end, out_list):
    cur = start
    while cur + pd.Timedelta(hours=2) <= max_end:
        end = cur + pd.Timedelta(hours=2)
        overlapped = insulin_df[
            (insulin_df["datetime"] >= cur)
            & (insulin_df["datetime"] < end)
            & (insulin_df["BWZ Carb Input (grams)"] > 0)
        ]
        if not overlapped.empty:
            cur = overlapped["datetime"].max() + pd.Timedelta(hours=2)
            continue
        idx24 = pd.date_range(cur, periods=24, freq="5min")
        seg24 = cgm_series.reindex(idx24).interpolate(limit_direction="both")
        if len(seg24) == 24 and seg24.isna().sum() <= 6:
            out_list.append(seg24.values)
        cur += FIVE_MIN

def extract_nomeal_24(insulin_df, cgm_df):
    insulin_df, cgm_series, meal_times, cmin, cmax = prep_frames(insulin_df, cgm_df)
    out = []
    if meal_times:
        _collect_nomeal(insulin_df, cgm_series, cmin, meal_times[0] - pd.Timedelta(minutes=30), out)
        for a, b in zip(meal_times[:-1], meal_times[1:]):
            _collect_nomeal(insulin_df, cgm_series, a + pd.Timedelta(hours=2), b - pd.Timedelta(minutes=30), out)
        _collect_nomeal(insulin_df, cgm_series, meal_times[-1] + pd.Timedelta(hours=2), cmax, out)
    return np.asarray(out)

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

def validate_model():
    print("Loading data...")
    ins1 = pd.read_csv("InsulinData.csv", low_memory=False)
    cgm1 = pd.read_csv("CGMData.csv", low_memory=False)
    ins2 = pd.read_csv("Insulin_patient2.csv", low_memory=False)
    cgm2 = pd.read_csv("CGM_patient2.csv", low_memory=False)

    print("Extracting meal and non-meal windows...")
    m1, n1 = extract_meal_24(ins1, cgm1), extract_nomeal_24(ins1, cgm1)
    m2, n2 = extract_meal_24(ins2, cgm2), extract_nomeal_24(ins2, cgm2)

    meal   = m1 if m2.size == 0 else (m2 if m1.size == 0 else np.vstack([m1, m2]))
    nomeal = n1 if n2.size == 0 else (n2 if n1.size == 0 else np.vstack([n1, n2]))

    print(f"Meal samples: {len(meal)}")
    print(f"Non-meal samples: {len(nomeal)}")

    if meal.size == 0 and nomeal.size == 0:
        X24 = np.zeros((2, 24)); y = np.array([1, 0], dtype=int)
    elif meal.size == 0 or nomeal.size == 0:
        X24 = meal if meal.size else nomeal
        y = np.ones(len(X24), dtype=int) if meal.size else np.zeros(len(X24), dtype=int)
    else:
        X24 = np.vstack([meal, nomeal])
        y = np.hstack([np.ones(len(meal), dtype=int), np.zeros(len(nomeal), dtype=int)])

    print(f"Total samples: {len(X24)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Improved balancing
    if len(np.unique(y)) == 2:
        idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
        if idx1.size and idx0.size and idx0.size > idx1.size * 4:
            sel0 = RNG.choice(idx0, size=min(int(2.0 * idx1.size), idx0.size), replace=False)
            keep = np.r_[idx1, sel0]
            X24, y = X24[keep], y[keep]
            print(f"After balancing - Total samples: {len(X24)}")
            print(f"After balancing - Class distribution: {np.bincount(y)}")

    print("Extracting features...")
    F = features_24(X24)
    print(f"Feature matrix shape: {F.shape}")

    # Load the trained model
    print("Loading trained model...")
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    thr = float(bundle.get("threshold", 0.5))
    print(f"Model threshold: {thr}")

    # Make predictions
    print("Making predictions...")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(F)[:, 1]
    else:
        s = model.decision_function(F)
        probs = (s - s.min()) / (s.max() - s.min() + 1e-12)
    
    y_pred = (probs >= thr).astype(int)

    # Calculate metrics
    print("\n=== VALIDATION RESULTS ===")
    print(f"Accuracy: {(y_pred == y).mean():.4f}")
    print(f"F1 Score: {f1_score(y, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print(f"\nThreshold used: {thr}")
    print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    # Show some sample predictions
    print("\nSample predictions (first 10):")
    for i in range(min(10, len(y))):
        print(f"Sample {i}: True={y[i]}, Pred={y_pred[i]}, Prob={probs[i]:.4f}")

if __name__ == "__main__":
    validate_model()