import pandas as pd
import numpy as np
import pickle

def optimized_meal_detector(glucose_data):
    """
    Optimized meal detector with refined thresholds and additional patterns
    """
    if len(glucose_data) < 24:
        return 0
    
    # Convert to numpy array
    glucose = np.array(glucose_data)
    
    # Calculate baseline (first 2 points average)
    baseline = np.mean(glucose[:2])
    
    # Rule 1: Early rise (15-30 minutes post-meal) - More sensitive
    if len(glucose) >= 6:
        early_rise = glucose[5] - baseline
        if early_rise > 7:  # Further reduced threshold
            return 1
    
    # Rule 2: Peak rise (45-90 minutes post-meal) - More sensitive
    if len(glucose) >= 12:
        peak_rise = np.max(glucose[:12]) - baseline
        if peak_rise > 13:  # Further reduced threshold
            return 1
    
    # Rule 3: Sustained elevation - More sensitive
    if len(glucose) >= 18:
        sustained_rise = glucose[17] - baseline
        if sustained_rise > 10:  # Further reduced threshold
            return 1
    
    # Rule 4: Rate of rise - More sensitive
    if len(glucose) >= 6:
        rates = np.diff(glucose[:6])
        avg_rate = np.mean(rates)
        if avg_rate > 1.2:  # Further reduced threshold
            return 1
    
    # Rule 5: Pattern consistency - Enhanced
    if len(glucose) >= 12:
        first_half = glucose[:6]
        second_half = glucose[6:12]
        
        if (np.max(second_half) > np.max(first_half) and 
            np.mean(second_half) > np.mean(first_half)):
            return 1
    
    # Rule 6: Gradual rise pattern - New rule
    if len(glucose) >= 8:
        # Check for gradual rise over first 8 points
        rise_pattern = glucose[7] - glucose[0]
        if rise_pattern > 10:  # Reduced threshold
            return 1
    
    # Rule 7: Mid-range elevation - New rule
    if len(glucose) >= 10:
        mid_rise = glucose[9] - baseline
        if mid_rise > 8:  # Reduced threshold
            return 1
    
    # Rule 8: Overall glucose range - New rule
    glucose_range = np.max(glucose) - np.min(glucose)
    if glucose_range > 20:  # Reduced threshold
        return 1
    
    # Rule 9: Positive slope dominance - New rule
    if len(glucose) >= 8:
        slopes = np.diff(glucose[:8])
        positive_slopes = np.sum(slopes > 0)
        if positive_slopes > len(slopes) * 0.55:  # Reduced threshold
            return 1
    
    # Rule 10: Late rise pattern - New rule
    if len(glucose) >= 16:
        late_rise = glucose[15] - glucose[8]  # Rise from 40 to 80 minutes
        if late_rise > 6:  # Reduced threshold
            return 1
    
    # Rule 11: Early acceleration - New rule
    if len(glucose) >= 4:
        early_accel = glucose[3] - glucose[0]
        if early_accel > 5:  # Early acceleration
            return 1
    
    # Rule 12: Steady rise pattern - New rule
    if len(glucose) >= 14:
        steady_rise = glucose[13] - glucose[6]  # Rise from 30 to 70 minutes
        if steady_rise > 8:  # Steady rise
            return 1
    
    # Rule 13: High baseline with rise - New rule
    if baseline > 140:  # High baseline
        if len(glucose) >= 6:
            if glucose[5] > baseline + 5:  # Rise from high baseline
                return 1
    
    # Rule 14: Multiple rise points - New rule
    if len(glucose) >= 12:
        rise_points = 0
        for i in range(1, 12):
            if glucose[i] > glucose[i-1]:
                rise_points += 1
        if rise_points >= 7:  # More than 7 rising points
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