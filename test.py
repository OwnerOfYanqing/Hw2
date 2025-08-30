import pandas as pd
import numpy as np
import pickle

def medical_meal_detector(glucose_data):
    """
    Medically-informed meal detector based on typical post-meal glucose patterns
    """
    if len(glucose_data) < 24:
        return 0
    
    # Convert to numpy array
    glucose = np.array(glucose_data)
    
    # Calculate baseline (first 2 points average)
    baseline = np.mean(glucose[:2])
    
    # Rule 1: Early rise (15-30 minutes post-meal)
    # Typical post-meal glucose starts rising within 15-30 minutes
    if len(glucose) >= 6:
        early_rise = glucose[5] - baseline
        if early_rise > 10:  # Rise of more than 10 mg/dL in 30 minutes
            return 1
    
    # Rule 2: Peak rise (45-90 minutes post-meal)
    # Post-meal glucose typically peaks between 45-90 minutes
    if len(glucose) >= 12:
        peak_rise = np.max(glucose[:12]) - baseline
        if peak_rise > 20:  # Peak rise of more than 20 mg/dL
            return 1
    
    # Rule 3: Sustained elevation
    # Post-meal glucose remains elevated for 1-2 hours
    if len(glucose) >= 18:
        sustained_rise = glucose[17] - baseline
        if sustained_rise > 15:  # Still elevated after 1.5 hours
            return 1
    
    # Rule 4: Rate of rise
    # Calculate average rate of rise in first 6 points
    if len(glucose) >= 6:
        rates = np.diff(glucose[:6])
        avg_rate = np.mean(rates)
        if avg_rate > 2:  # Average rate of rise > 2 mg/dL per 5 minutes
            return 1
    
    # Rule 5: Pattern consistency
    # Check if the pattern looks like a typical meal response
    if len(glucose) >= 12:
        # Look for a rise followed by a peak
        first_half = glucose[:6]
        second_half = glucose[6:12]
        
        if (np.max(second_half) > np.max(first_half) and 
            np.mean(second_half) > np.mean(first_half)):
            return 1
    
    return 0

def main():
    # 1) 读取模型
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    
    model_type = bundle.get("model_type", "medical_rule")
    
    # 2) 读取测试矩阵 N×24（无表头）
    test_df = pd.read_csv("test.csv", header=None)
    X24 = test_df.values.astype(float)

    # 3) 使用医学检测器进行预测
    y_pred = []
    for i in range(len(X24)):
        glucose_window = X24[i]
        pred = medical_meal_detector(glucose_window)
        y_pred.append(pred)

    # 4) 输出单列、无表头，与 test.csv 行数一致
    pd.Series(y_pred).to_csv("Result.csv", index=False, header=False)

if __name__ == "__main__":
    main()