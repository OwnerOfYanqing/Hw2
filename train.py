import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif

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
    
    # Filter for significant carb inputs (>= 10g to reduce noise)
    meal_times = list(
        insulin_df[insulin_df["BWZ Carb Input (grams)"] >= 10]["datetime"]
        .sort_values()
        .unique()
    )
    return insulin_df, cgm_series, meal_times, cgm_df["datetime"].min(), cgm_df["datetime"].max()

# ---------- improved window extraction ----------
def get_cgm_window_robust(cgm_series, start_time, duration_minutes=120, target_points=24):
    """More robust CGM window extraction"""
    end_time = start_time + pd.Timedelta(minutes=duration_minutes)
    window_data = cgm_series.loc[start_time:end_time].dropna()
    
    if len(window_data) < target_points // 2:
        return None
    
    if len(window_data) >= target_points * 0.8:
        resampled = window_data.resample('5min').mean().interpolate(limit_direction='both')
        
        if len(resampled) >= target_points:
            return resampled.iloc[:target_points].values
        elif len(resampled) >= target_points * 0.8:
            time_index = pd.date_range(start_time, end_time, periods=target_points)
            all_data = pd.concat([window_data, pd.Series(index=time_index, dtype=float)])
            all_data = all_data.groupby(all_data.index).first().sort_index()
            all_data = all_data.interpolate(method='time', limit_direction='both')
            result = all_data.reindex(time_index)
            
            if result.isna().sum() <= target_points // 3:
                return result.ffill().bfill().values
    return None

def extract_meal_24(insulin_df, cgm_df):
    insulin_df, cgm_series, meal_times, cmin, cmax = prep_frames(insulin_df, cgm_df)
    out = []
    
    valid_meals = [tm for tm in meal_times if cmin + pd.Timedelta(hours=1) <= tm <= cmax - pd.Timedelta(hours=2)]
    
    i = 0
    while i < len(valid_meals):
        tm = valid_meals[i]
        j = i + 1
        picked = tm
        while j < len(valid_meals) and (valid_meals[j] - picked) < pd.Timedelta(hours=2) - TOL:
            picked = valid_meals[j]
            j += 1
        
        start = picked - pd.Timedelta(minutes=30)
        window = get_cgm_window_robust(cgm_series, start, duration_minutes=120, target_points=24)
        
        if window is not None:
            out.append(window)
        
        i = j if j > i else i + 1
    
    return np.asarray(out)



def extract_nomeal_24(insulin_df, cgm_df):
    insulin_df, cgm_series, meal_times, cmin, cmax = prep_frames(insulin_df, cgm_df)
    out = []
    
    valid_meals = [tm for tm in meal_times if cmin <= tm <= cmax]
    
    if not valid_meals:
        current = cmin + pd.Timedelta(hours=1)
        while current + pd.Timedelta(hours=2) <= cmax - pd.Timedelta(hours=1):
            window = get_cgm_window_robust(cgm_series, current, duration_minutes=120, target_points=24)
            if window is not None:
                out.append(window)
            current += pd.Timedelta(hours=4)
    else:
        # Sample between meals with larger buffers
        for i in range(len(valid_meals) - 1):
            meal_end = valid_meals[i] + pd.Timedelta(hours=3)
            next_meal_start = valid_meals[i + 1] - pd.Timedelta(hours=3)
            
            if next_meal_start > meal_end + pd.Timedelta(hours=3):
                current = meal_end
                while current + pd.Timedelta(hours=2) < next_meal_start:
                    window = get_cgm_window_robust(cgm_series, current, duration_minutes=120, target_points=24)
                    if window is not None:
                        out.append(window)
                    current += pd.Timedelta(hours=4)
    
    return np.asarray(out)

# ---------- features (shared) ----------
def features_24(M):
    feats = []
    for row in M:
        row = np.nan_to_num(row.astype(float))
        baseline = float(np.mean(row))
        x = row - baseline

        mean = float(np.mean(x))
        std = float(np.std(x))
        vmin = float(np.min(x))
        vmax = float(np.max(x))
        rge = vmax - vmin
        rms = float(np.sqrt(np.mean(x**2)))
        skw = float(skew(x, bias=False)) if std > 1e-12 else 0.0
        kur = float(kurtosis(x, bias=False)) if std > 1e-12 else 0.0

        dx = np.diff(x)
        dv_mean = float(np.mean(dx)) if dx.size else 0.0
        dv_std  = float(np.std(dx))  if dx.size else 0.0
        dv_max  = float(np.max(dx))  if dx.size else 0.0
        dv_min  = float(np.min(dx))  if dx.size else 0.0
        ddx = np.diff(dx)
        da_std = float(np.std(ddx)) if ddx.size else 0.0

        try:
            auc = float(np.trapezoid(x))
        except Exception:
            auc = float(np.trapz(x))

        peaks, _ = find_peaks(x)
        num_peaks = int(len(peaks))
        t_max = int(np.argmax(x))
        pos_sum = float(np.sum(dx[dx > 0])) if dx.size else 0.0

        max_slope30 = 0.0
        for k in range(0, len(x) - 6):
            max_slope30 = max(max_slope30, (x[k + 6] - x[k]) / 6.0)

        fft_mag = np.abs(np.fft.fft(x))
        if fft_mag.size < 7:
            fft_feats = np.pad(fft_mag[1:], (0, 6 - max(0, fft_mag.size - 1)))
        else:
            fft_feats = fft_mag[1:7]

        feat = [
            baseline, mean, std, vmin, vmax, rge, rms, skw, kur,
            dv_mean, dv_std, dv_max, dv_min, da_std,
            auc, num_peaks, t_max, pos_sum, max_slope30,
            float(fft_feats[0]), float(fft_feats[1]), float(fft_feats[2]),
            float(fft_feats[3]), float(fft_feats[4]), float(fft_feats[5])
        ]
        feats.append(np.nan_to_num(feat))
    return np.asarray(feats, dtype=float)

# ---------- threshold search ----------
def _grid_thr_from_probs(y, probs):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.15, 0.8501, 0.001):  # Wider range, finer grid
        f1 = f1_score(y, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    return best_thr

def safe_best_threshold(model, X, y):
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    counts = np.bincount(y)
    min_class = int(min(counts[0], counts[1]))
    if min_class < 2:
        mdl = clone(model).fit(X, y)
        probs = mdl.predict_proba(X)[:, 1]
        return _grid_thr_from_probs(y, probs)
    n_splits = min(5, min_class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = np.zeros_like(y, dtype=float)
    for tr, va in skf.split(X, y):
        mdl = clone(model).fit(X[tr], y[tr])
        probs[va] = mdl.predict_proba(X[va])[:, 1]
    return _grid_thr_from_probs(y, probs)

# ---------- train ----------
def train_model():
    ins1 = pd.read_csv("InsulinData.csv", low_memory=False)
    cgm1 = pd.read_csv("CGMData.csv", low_memory=False)
    ins2 = pd.read_csv("Insulin_patient2.csv", low_memory=False)
    cgm2 = pd.read_csv("CGM_patient2.csv", low_memory=False)

    m1, n1 = extract_meal_24(ins1, cgm1), extract_nomeal_24(ins1, cgm1)
    m2, n2 = extract_meal_24(ins2, cgm2), extract_nomeal_24(ins2, cgm2)

    meal   = m1 if m2.size == 0 else (m2 if m1.size == 0 else np.vstack([m1, m2]))
    nomeal = n1 if n2.size == 0 else (n2 if n1.size == 0 else np.vstack([n1, n2]))

    if meal.size == 0 and nomeal.size == 0:
        X24 = np.zeros((2, 24)); y = np.array([1, 0], dtype=int)
    elif meal.size == 0 or nomeal.size == 0:
        X24 = meal if meal.size else nomeal
        y = np.ones(len(X24), dtype=int) if meal.size else np.zeros(len(X24), dtype=int)
    else:
        X24 = np.vstack([meal, nomeal])
        y = np.hstack([np.ones(len(meal), dtype=int), np.zeros(len(nomeal), dtype=int)])

    # Better balancing strategy
    if len(np.unique(y)) == 2:
        idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
        if idx1.size and idx0.size:
            # Aim for 1:1.5 ratio (meal:no-meal)
            if idx0.size > idx1.size * 1.5:
                target_nomeal = int(idx1.size * 1.5)
                sel0 = RNG.choice(idx0, size=target_nomeal, replace=False)
                keep = np.r_[idx1, sel0]
                X24, y = X24[keep], y[keep]

    F = features_24(X24)

    if len(np.unique(y)) < 2:
        model = Pipeline([("scaler", StandardScaler()),
                          ("clf", DummyClassifier(strategy="most_frequent"))])
        model.fit(F, y)
        thr = 0.5
        selector = None
    else:
        counts = np.bincount(y)
        min_class = int(min(counts[0], counts[1]))

        # Feature selection to reduce overfitting
        selector = SelectKBest(f_classif, k=min(25, F.shape[1]))
        F_selected = selector.fit_transform(F, y)

        svc_base = Pipeline([
            ("scaler", RobustScaler()),
            ("svc", SVC(C=5.0, kernel="rbf", gamma="auto",
                        class_weight="balanced", probability=True, random_state=42))
        ])
        rf_base = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_depth=12,
            class_weight="balanced", n_jobs=-1, random_state=42
        )
        gb_base = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            random_state=42
        )

        if min_class >= 3:
            svc = CalibratedClassifierCV(estimator=svc_base, method="isotonic", cv=3)
            rf  = CalibratedClassifierCV(estimator=rf_base,  method="isotonic", cv=3)
            gb  = CalibratedClassifierCV(estimator=gb_base,  method="isotonic", cv=3)
        elif min_class >= 2:
            svc = CalibratedClassifierCV(estimator=svc_base, method="isotonic", cv=2)
            rf  = CalibratedClassifierCV(estimator=rf_base,  method="isotonic", cv=2)
            gb  = CalibratedClassifierCV(estimator=gb_base,  method="isotonic", cv=2)
        else:
            svc, rf, gb = svc_base, rf_base, gb_base

        model = VotingClassifier(
            estimators=[("svc", svc), ("rf", rf), ("gb", gb)],
            voting="soft", weights=[1.5, 2.0, 1.0]
        )
        model.fit(F_selected, y)
        thr = safe_best_threshold(model, F_selected, y)

    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model": model, 
            "threshold": float(thr),
            "feature_selector": selector
        }, f)

if __name__ == "__main__":
    train_model()