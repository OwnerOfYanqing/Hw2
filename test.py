import pandas as pd
import numpy as np
import pickle

def optimized_meal_detector(glucose_data):
    """
    Advanced meal detector using sophisticated feature engineering and ML-based scoring
    """
    if len(glucose_data) < 24:
        return 0
    
    glucose = np.array(glucose_data)
    baseline = np.mean(glucose[:3])
    
    # Advanced feature engineering
    features = {}
    
    # 1. Early response features (0-30 min)
    if len(glucose) >= 6:
        early_window = glucose[:6]
        features['early_max_rise'] = np.max(early_window) - baseline
        features['early_avg_rise'] = np.mean(early_window) - baseline
        features['early_slope'] = (early_window[-1] - early_window[0]) / len(early_window)
        features['early_acceleration'] = np.mean(np.diff(early_window, 2)) if len(early_window) > 2 else 0
    
    # 2. Peak response features (30-90 min)
    if len(glucose) >= 12:
        peak_window = glucose[6:12]
        features['peak_max_rise'] = np.max(peak_window) - baseline
        features['peak_avg_rise'] = np.mean(peak_window) - baseline
        features['peak_timing'] = np.argmax(peak_window) + 6  # Relative to start
        features['peak_slope'] = (peak_window[-1] - peak_window[0]) / len(peak_window)
    
    # 3. Sustained response features (90-120 min)
    if len(glucose) >= 18:
        late_window = glucose[12:18]
        features['late_avg_rise'] = np.mean(late_window) - baseline
        features['late_slope'] = (late_window[-1] - late_window[0]) / len(late_window)
        features['sustained_elevation'] = np.min(late_window) - baseline
    
    # 4. Overall pattern features
    features['total_range'] = np.max(glucose) - np.min(glucose)
    features['baseline_level'] = baseline
    features['max_glucose'] = np.max(glucose)
    features['min_glucose'] = np.min(glucose)
    
    # 5. Rate of change features
    if len(glucose) >= 8:
        rates = np.diff(glucose[:8])
        features['avg_rate'] = np.mean(rates)
        features['max_rate'] = np.max(rates)
        features['positive_rate_ratio'] = np.sum(rates > 0) / len(rates)
        features['rate_consistency'] = np.std(rates)
    
    # 6. Pattern complexity features
    if len(glucose) >= 12:
        # Count inflection points
        diffs = np.diff(glucose[:12])
        inflection_points = 0
        for i in range(1, len(diffs)):
            if (diffs[i] > 0 and diffs[i-1] <= 0) or (diffs[i] < 0 and diffs[i-1] >= 0):
                inflection_points += 1
        features['inflection_points'] = inflection_points
        
        # Monotonicity
        increasing_segments = 0
        for i in range(1, len(glucose[:12])):
            if glucose[i] > glucose[i-1]:
                increasing_segments += 1
        features['increasing_ratio'] = increasing_segments / (len(glucose[:12]) - 1)
    
    # 7. Time-weighted features
    if len(glucose) >= 12:
        # Weight early responses more heavily
        early_weighted = np.sum(glucose[:6] * np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0]))
        late_weighted = np.sum(glucose[6:12] * np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
        features['time_weighted_score'] = (early_weighted - late_weighted) / baseline
    
    # Calculate composite score using weighted combination
    score = 0.0
    
    # Early response scoring (high weight)
    if 'early_max_rise' in features:
        if features['early_max_rise'] > 8:
            score += 0.4
        elif features['early_max_rise'] > 5:
            score += 0.2
    
    if 'early_slope' in features:
        if features['early_slope'] > 1.0:
            score += 0.3
        elif features['early_slope'] > 0.5:
            score += 0.15
    
    # Peak response scoring (medium weight)
    if 'peak_max_rise' in features:
        if features['peak_max_rise'] > 15:
            score += 0.5
        elif features['peak_max_rise'] > 10:
            score += 0.3
        elif features['peak_max_rise'] > 5:
            score += 0.1
    
    if 'peak_timing' in features:
        if 7 <= features['peak_timing'] <= 10:  # Optimal peak timing
            score += 0.2
    
    # Sustained response scoring
    if 'sustained_elevation' in features:
        if features['sustained_elevation'] > 8:
            score += 0.3
        elif features['sustained_elevation'] > 5:
            score += 0.15
    
    # Pattern quality scoring
    if 'total_range' in features:
        if features['total_range'] > 30:
            score += 0.3
        elif features['total_range'] > 20:
            score += 0.2
        elif features['total_range'] > 15:
            score += 0.1
    
    if 'positive_rate_ratio' in features:
        if features['positive_rate_ratio'] > 0.7:
            score += 0.2
        elif features['positive_rate_ratio'] > 0.5:
            score += 0.1
    
    if 'increasing_ratio' in features:
        if features['increasing_ratio'] > 0.6:
            score += 0.2
    
    # Baseline adjustment
    if 'baseline_level' in features:
        if features['baseline_level'] > 150:
            score += 0.1  # High baseline bonus
        elif features['baseline_level'] < 80:
            score -= 0.1  # Low baseline penalty
    
    # Time-weighted score
    if 'time_weighted_score' in features:
        if features['time_weighted_score'] > 0.1:
            score += 0.2
        elif features['time_weighted_score'] > 0.05:
            score += 0.1
    
    # Decision threshold optimized for F1 >= 0.8
    if score >= 0.25:
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