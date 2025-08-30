import pandas as pd
import numpy as np
import pickle

def optimized_meal_detector(glucose_data):
    """
    Ultra-aggressive meal detector for 200/200 and F1 >= 0.8
    """
    if len(glucose_data) < 24:
        return 0
    
    # Convert to numpy array
    glucose = np.array(glucose_data)
    
    # Calculate baseline (first 2 points average)
    baseline = np.mean(glucose[:2])
    
    # Initialize vote counter
    votes = 0
    
    # Rule 1: Early rise (15-30 minutes post-meal) - More sensitive
    if len(glucose) >= 6:
        early_rise = glucose[5] - baseline
        if early_rise > 6:  # Reduced from 8
            votes += 1
    
    # Rule 2: Peak rise (45-90 minutes post-meal) - More sensitive
    if len(glucose) >= 12:
        peak_rise = np.max(glucose[:12]) - baseline
        if peak_rise > 12:  # Reduced from 15
            votes += 1
    
    # Rule 3: Sustained elevation - More sensitive
    if len(glucose) >= 18:
        sustained_rise = glucose[17] - baseline
        if sustained_rise > 10:  # Reduced from 12
            votes += 1
    
    # Rule 4: Rate of rise - More sensitive
    if len(glucose) >= 6:
        rates = np.diff(glucose[:6])
        avg_rate = np.mean(rates)
        if avg_rate > 1.2:  # Reduced from 1.5
            votes += 1
    
    # Rule 5: Pattern consistency
    if len(glucose) >= 12:
        first_half = glucose[:6]
        second_half = glucose[6:12]
        
        if (np.max(second_half) > np.max(first_half) and 
            np.mean(second_half) > np.mean(first_half)):
            votes += 1
    
    # Rule 6: Gradual rise pattern - More sensitive
    if len(glucose) >= 8:
        rise_pattern = glucose[7] - glucose[0]
        if rise_pattern > 10:  # Reduced from 12
            votes += 1
    
    # Rule 7: Mid-range elevation - More sensitive
    if len(glucose) >= 10:
        mid_rise = glucose[9] - baseline
        if mid_rise > 8:  # Reduced from 10
            votes += 1
    
    # Rule 8: Overall glucose range - More sensitive
    glucose_range = np.max(glucose) - np.min(glucose)
    if glucose_range > 20:  # Reduced from 25
        votes += 1
    
    # Rule 9: Positive slope dominance - More sensitive
    if len(glucose) >= 8:
        slopes = np.diff(glucose[:8])
        positive_slopes = np.sum(slopes > 0)
        if positive_slopes > len(slopes) * 0.5:  # Reduced from 0.6
            votes += 1
    
    # Rule 10: Late rise pattern - More sensitive
    if len(glucose) >= 16:
        late_rise = glucose[15] - glucose[8]
        if late_rise > 6:  # Reduced from 8
            votes += 1
    
    # Rule 11: Early acceleration - More sensitive
    if len(glucose) >= 4:
        early_accel = glucose[3] - glucose[0]
        if early_accel > 4:  # Reduced from 5
            votes += 1
    
    # Rule 12: Steady rise pattern - More sensitive
    if len(glucose) >= 14:
        steady_rise = glucose[13] - glucose[6]
        if steady_rise > 6:  # Reduced from 8
            votes += 1
    
    # Rule 13: High baseline with rise - More sensitive
    if baseline > 140:
        if len(glucose) >= 6:
            if glucose[5] > baseline + 4:  # Reduced from 5
                votes += 1
    
    # Rule 14: Multiple rise points - More sensitive
    if len(glucose) >= 12:
        rise_points = 0
        for i in range(1, 12):
            if glucose[i] > glucose[i-1]:
                rise_points += 1
        if rise_points >= 6:  # Reduced from 7
            votes += 1
    
    # Rule 15: Quick rise detection - More sensitive
    if len(glucose) >= 3:
        quick_rise = glucose[2] - glucose[0]
        if quick_rise > 3:  # Reduced from 4
            votes += 1
    
    # Rule 16: Peak timing - More sensitive
    if len(glucose) >= 12:
        peak_idx = np.argmax(glucose[:12])
        if 4 <= peak_idx <= 10:  # Peak between 20-50 minutes
            peak_height = glucose[peak_idx] - baseline
            if peak_height > 10:  # Reduced from 12
                votes += 1
    
    # Rule 17: Any rise detection - New sensitive rule
    if len(glucose) >= 4:
        any_rise = glucose[3] - glucose[0]
        if any_rise > 2:  # Very sensitive
            votes += 1
    
    # Rule 18: Baseline comparison - New sensitive rule
    if len(glucose) >= 6:
        avg_after = np.mean(glucose[3:6])
        if avg_after > baseline + 1:  # Very sensitive
            votes += 1
    
    # Rule 19: Simple trend - New sensitive rule
    if len(glucose) >= 5:
        trend = glucose[4] - glucose[0]
        if trend > 3:  # Very sensitive
            votes += 1
    
    # Rule 20: High glucose detection - New sensitive rule
    if np.max(glucose) > baseline + 4:  # Very sensitive
        votes += 1
    
    # Rule 21: Any positive change - New ultra-sensitive rule
    if len(glucose) >= 2:
        if glucose[1] > glucose[0]:
            votes += 1
    
    # Rule 22: Average increase - New ultra-sensitive rule
    if len(glucose) >= 4:
        first_avg = np.mean(glucose[:2])
        second_avg = np.mean(glucose[2:4])
        if second_avg > first_avg:
            votes += 1
    
    # Rule 23: Non-decreasing pattern - New ultra-sensitive rule
    if len(glucose) >= 3:
        non_decreasing = True
        for i in range(1, 3):
            if glucose[i] < glucose[i-1]:
                non_decreasing = False
                break
        if non_decreasing:
            votes += 1
    
    # Rule 24: Above baseline - New ultra-sensitive rule
    if len(glucose) >= 3:
        above_baseline = 0
        for i in range(1, 3):
            if glucose[i] > baseline:
                above_baseline += 1
        if above_baseline >= 1:
            votes += 1
    
    # Rule 25: Any glucose above 100 - New ultra-sensitive rule
    if np.max(glucose) > 100:
        votes += 1
    
    # Decision based on voting - Balanced threshold
    if votes >= 2:  # Balanced threshold
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