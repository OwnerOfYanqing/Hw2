import pandas as pd
import numpy as np
import pickle

def optimized_meal_detector(glucose_data):
    """
    Final optimized meal detector based on best performing approach
    """
    if len(glucose_data) < 24:
        return 0
    
    glucose = np.array(glucose_data)
    baseline = np.mean(glucose[:3])
    
    # Feature extraction
    early_rise = 0
    early_slope = 0
    mid_rise = 0
    total_range = 0
    positive_rate_ratio = 0
    
    if len(glucose) >= 6:
        early_window = glucose[:6]
        early_rise = np.max(early_window) - baseline
        early_slope = (early_window[-1] - early_window[0]) / 5
    
    if len(glucose) >= 12:
        mid_window = glucose[6:12]
        mid_rise = np.max(mid_window) - baseline
    
    total_range = np.max(glucose) - np.min(glucose)
    
    if len(glucose) >= 8:
        rates = np.diff(glucose[:8])
        positive_rate_ratio = np.sum(rates > 0) / len(rates)
    
    # Scoring system optimized for F1 >= 0.8
    score = 0.0
    
    # Early response scoring (optimized thresholds)
    if early_rise > 4:
        score += 0.5
    elif early_rise > 2:
        score += 0.3
    elif early_rise > 0.5:
        score += 0.1
    
    if early_slope > 0.8:
        score += 0.4
    elif early_slope > 0.3:
        score += 0.2
    elif early_slope > 0.1:
        score += 0.1
    
    # Mid response scoring (optimized thresholds)
    if mid_rise > 10:
        score += 0.6
    elif mid_rise > 6:
        score += 0.4
    elif mid_rise > 3:
        score += 0.2
    elif mid_rise > 1:
        score += 0.1
    
    # Pattern scoring (optimized thresholds)
    if total_range > 20:
        score += 0.4
    elif total_range > 12:
        score += 0.3
    elif total_range > 6:
        score += 0.2
    elif total_range > 2:
        score += 0.1
    
    if positive_rate_ratio > 0.6:
        score += 0.3
    elif positive_rate_ratio > 0.4:
        score += 0.2
    elif positive_rate_ratio > 0.2:
        score += 0.1
    
    # Glucose level considerations
    max_glucose = np.max(glucose)
    if max_glucose > 170:
        score += 0.2
    elif max_glucose > 150:
        score += 0.1
    
    # Baseline considerations
    if baseline > 130:
        score += 0.1
    elif baseline < 50:
        score -= 0.1
    
    # Optimized decision threshold for F1 >= 0.8
    if score >= 0.8:
        return 1
    
    # Fallback conditions for borderline cases
    if score >= 0.4:
        if early_rise > 1.5:
            return 1
        if mid_rise > 4:
            return 1
        if total_range > 8:
            return 1
        if positive_rate_ratio > 0.3:
            return 1
    
    # Additional lenient checks
    if score >= 0.2:
        if early_rise > 0.5 and mid_rise > 2:
            return 1
        if total_range > 5 and positive_rate_ratio > 0.2:
            return 1
    
    return 0

def main():
    # 1) 读取模型
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    
    model_type = bundle.get("model_type", "optimized_rule")
    
    # 2) 读取测试矩阵 N×24（无表头）
    test_df = pd.read_csv("test.csv", header=None)
    X24 = test_df.values.astype(float)

    # 3) 使用优化检测器进行预测
    y_pred = []
    for i in range(len(X24)):
        glucose_window = X24[i]
        pred = optimized_meal_detector(glucose_window)
        y_pred.append(pred)

    # 4) 输出单列、无表头，与 test.csv 行数一致
    pd.Series(y_pred).to_csv("Result.csv", index=False, header=False)

if __name__ == "__main__":
    main()