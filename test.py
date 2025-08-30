import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

np.seterr(all="ignore")

# --- features: 与 train.py 完全一致 ---
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

def main():
    # 1) 读取模型
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    thr = float(bundle.get("threshold", 0.5))
    selector = bundle.get("feature_selector", None)

    # 2) 读取测试矩阵 N×24（无表头）
    test_df = pd.read_csv("test.csv", header=None)
    X24 = test_df.values.astype(float)

    # 3) 提取特征并预测
    F = features_24(X24)
    
    # Apply feature selection if available
    if selector is not None:
        F = selector.transform(F)
    
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