import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.dummy import DummyClassifier


FIVE_MIN = pd.Timedelta(minutes=5)
TWO_HOURS = pd.Timedelta(hours=2)
HALF_HOUR = pd.Timedelta(minutes=30)
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


def prep_frames(insulin_df: pd.DataFrame, cgm_df: pd.DataFrame):
    insulin_df = insulin_df.copy()
    cgm_df = cgm_df.copy()

    # Normalize and build datetime
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
    cgm_df = (
        cgm_df.dropna(subset=["datetime"])\
              .sort_values("datetime")\
              .drop_duplicates(subset=["datetime"], keep="last")
    )

    cgm_series = cgm_df.set_index("datetime")["Sensor Glucose (mg/dL)"].sort_index()

    meal_times = list(
        insulin_df[insulin_df["BWZ Carb Input (grams)"] > 0]["datetime"]
        .sort_values()
        .unique()
    )

    return (
        insulin_df,
        cgm_series,
        meal_times,
        cgm_df["datetime"].min(),
        cgm_df["datetime"].max(),
    )


# ---------- window extraction (train only) ----------
def extract_meal_24(insulin_df: pd.DataFrame, cgm_df: pd.DataFrame) -> np.ndarray:
    """Extract meal windows as 24 points at 5-min intervals covering [-30, +90] minutes
    around the final event in any cluster of meals that are within 2 hours of each other.
    """
    insulin_df, cgm_series, meal_times, _, _ = prep_frames(insulin_df, cgm_df)
    out = []
    i = 0
    while i < len(meal_times):
        tm = meal_times[i]
        # Collapse events within 2 hours into a cluster; keep the last time in the cluster
        j = i + 1
        picked = tm
        while j < len(meal_times) and (meal_times[j] - picked) < (TWO_HOURS - TOL):
            picked = meal_times[j]
            j += 1

        # Standard window aligned to [-30, +90] around the picked event
        start = picked - HALF_HOUR
        idx24 = pd.date_range(start, periods=24, freq="5min")
        seg24 = cgm_series.reindex(idx24).interpolate(method="time", limit_direction="both")
        if len(seg24) == 24 and seg24.isna().sum() <= 6:
            out.append(seg24.values.astype(float))
        i = j
    return np.asarray(out)


def _collect_nomeal(insulin_df: pd.DataFrame, cgm_series: pd.Series,
                    start: pd.Timestamp, max_end: pd.Timestamp, out_list: list):
    cur = start
    while cur + TWO_HOURS <= max_end:
        end = cur + TWO_HOURS
        overlapped = insulin_df[
            (insulin_df["datetime"] >= cur)
            & (insulin_df["datetime"] < end)
            & (insulin_df["BWZ Carb Input (grams)"] > 0)
        ]
        if not overlapped.empty:
            # Jump to 2 hours after the last overlapping event
            cur = overlapped["datetime"].max() + TWO_HOURS
            continue

        idx24 = pd.date_range(cur, periods=24, freq="5min")
        seg24 = cgm_series.reindex(idx24).interpolate(method="time", limit_direction="both")
        if len(seg24) == 24 and seg24.isna().sum() <= 6:
            out_list.append(seg24.values.astype(float))
        cur += FIVE_MIN


def extract_nomeal_24(insulin_df: pd.DataFrame, cgm_df: pd.DataFrame) -> np.ndarray:
    insulin_df, cgm_series, meal_times, cmin, cmax = prep_frames(insulin_df, cgm_df)
    out = []
    if meal_times:
        _collect_nomeal(insulin_df, cgm_series, cmin, meal_times[0] - HALF_HOUR, out)
        for a, b in zip(meal_times[:-1], meal_times[1:]):
            _collect_nomeal(insulin_df, cgm_series, a + TWO_HOURS, b - HALF_HOUR, out)
        _collect_nomeal(insulin_df, cgm_series, meal_times[-1] + TWO_HOURS, cmax, out)
    return np.asarray(out)


# ---------- features (shared) ----------
def features_24(M: np.ndarray) -> np.ndarray:
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
            fft_feats = np.pad(fft_mag[1:], (0, max(0, 6 - max(0, fft_mag.size - 1))))
        else:
            fft_feats = fft_mag[1:7]

        feat = [
            baseline, mean, std, vmin, vmax, rge, rms, skw, kur,
            dv_mean, dv_std, dv_max, dv_min, da_std,
            auc, num_peaks, t_max, pos_sum, max_slope30,
            float(fft_feats[0]) if fft_feats.size > 0 else 0.0,
            float(fft_feats[1]) if fft_feats.size > 1 else 0.0,
            float(fft_feats[2]) if fft_feats.size > 2 else 0.0,
            float(fft_feats[3]) if fft_feats.size > 3 else 0.0,
            float(fft_feats[4]) if fft_feats.size > 4 else 0.0,
            float(fft_feats[5]) if fft_feats.size > 5 else 0.0,
        ]
        feats.append(np.nan_to_num(feat))
    return np.asarray(feats, dtype=float)


# ---------- threshold search ----------
def _grid_thr_from_probs(y: np.ndarray, probs: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.20, 0.8001, 0.002):
        f1 = f1_score(y, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    return best_thr


def safe_best_threshold(model, X: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    counts = np.bincount(y)
    if counts.size < 2:
        return 0.5
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


def _build_model(y: np.ndarray):
    if len(np.unique(y)) < 2:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ])

    counts = np.bincount(y)
    min_class = int(min(counts[0], counts[1]))

    svc_base = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=8.5, kernel="rbf", gamma="scale",
                     class_weight="balanced", probability=True, random_state=42)),
    ])
    rf_base = RandomForestClassifier(
        n_estimators=600, min_samples_leaf=1, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )

    if min_class >= 3:
        svc = CalibratedClassifierCV(estimator=svc_base, method="sigmoid", cv=3)
        rf = CalibratedClassifierCV(estimator=rf_base, method="sigmoid", cv=3)
    elif min_class >= 2:
        svc = CalibratedClassifierCV(estimator=svc_base, method="sigmoid", cv=2)
        rf = CalibratedClassifierCV(estimator=rf_base, method="sigmoid", cv=2)
    else:
        svc, rf = svc_base, rf_base

    model = VotingClassifier(estimators=[("svc", svc), ("rf", rf)],
                             voting="soft", weights=[2.6, 1.0])
    return model


def _evaluate_cv(model, F: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(y) < 10:
        return float("nan")
    skf = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=42)
    f1s = []
    for tr, va in skf.split(F, y):
        mdl = clone(model)
        mdl.fit(F[tr], y[tr])
        thr_tr = safe_best_threshold(mdl, F[tr], y[tr])
        probs = mdl.predict_proba(F[va])[:, 1]
        pred = (probs >= thr_tr).astype(int)
        f1s.append(f1_score(y[va], pred, zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan")


# ---------- train ----------
def train_model():
    # 1) Load datasets
    ins1 = pd.read_csv("/workspace/InsulinData.csv", low_memory=False)
    cgm1 = pd.read_csv("/workspace/CGMData.csv", low_memory=False)
    ins2 = pd.read_csv("/workspace/Insulin_patient2.csv", low_memory=False)
    cgm2 = pd.read_csv("/workspace/CGM_patient2.csv", low_memory=False)

    # 2) Extract windows
    m1, n1 = extract_meal_24(ins1, cgm1), extract_nomeal_24(ins1, cgm1)
    m2, n2 = extract_meal_24(ins2, cgm2), extract_nomeal_24(ins2, cgm2)

    meal = m1 if m2.size == 0 else (m2 if m1.size == 0 else np.vstack([m1, m2]))
    nomeal = n1 if n2.size == 0 else (n2 if n1.size == 0 else np.vstack([n1, n2]))

    # 3) Build X, y
    if meal.size == 0 and nomeal.size == 0:
        X24 = np.zeros((2, 24)); y = np.array([1, 0], dtype=int)
    elif meal.size == 0 or nomeal.size == 0:
        X24 = meal if meal.size else nomeal
        y = np.ones(len(X24), dtype=int) if meal.size else np.zeros(len(X24), dtype=int)
    else:
        X24 = np.vstack([meal, nomeal])
        y = np.hstack([np.ones(len(meal), dtype=int), np.zeros(len(nomeal), dtype=int)])

    # Light downsampling to mitigate extreme imbalance
    if len(np.unique(y)) == 2:
        idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
        if idx1.size and idx0.size and idx0.size > idx1.size * 3:
            sel0 = RNG.choice(idx0, size=min(int(1.5 * idx1.size), idx0.size), replace=False)
            keep = np.r_[idx1, sel0]
            X24, y = X24[keep], y[keep]

    # 4) Feature engineering
    F = features_24(X24)

    # 5) Modeling
    model = _build_model(y)
    model.fit(F, y)
    thr = safe_best_threshold(model, F, y)

    # 6) Cross-validated estimate (for verification)
    cv_f1 = _evaluate_cv(model, F, y)
    print(f"Meal windows: {len(meal)} | No-meal windows: {len(nomeal)} | Total: {len(y)}")
    print(f"CV F1 (mean): {cv_f1:.4f}")
    print(f"Chosen threshold: {thr:.3f}")

    # 7) Persist model
    with open("/workspace/model.pkl", "wb") as f:
        pickle.dump({"model": model, "threshold": float(thr)}, f)


if __name__ == "__main__":
    train_model()

