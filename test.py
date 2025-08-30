import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft

np.seterr(all="ignore")

# --- enhanced features: matching train.py ---
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
    # 1) 读取模型
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    thr = float(bundle.get("threshold", 0.5))

    # 2) 读取测试矩阵 N×24（无表头）
    test_df = pd.read_csv("test.csv", header=None)
    X24 = test_df.values.astype(float)

    # 3) 提特征并预测
    F = features_24(X24)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(F)[:, 1]
    else:
        s = model.decision_function(F)
        probs = (s - s.min()) / (s.max() - s.min() + 1e-12)
    y_pred = (probs >= thr).astype(int)

    # 4) 输出单列、无表头，与 test.csv 行数一致
    pd.Series(y_pred.astype(int)).to_csv("Result.csv", index=False, header=False)

if __name__ == "__main__":
    main()