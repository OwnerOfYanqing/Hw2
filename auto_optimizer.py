import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score

def create_detector(thresholds, voting_threshold):
    """
    Create a meal detector with given thresholds
    """
    def detector(glucose_data):
        if len(glucose_data) < 24:
            return 0
        
        glucose = np.array(glucose_data)
        baseline = np.mean(glucose[:2])
        votes = 0
        
        # Rule 1: Early rise
        if len(glucose) >= 6:
            early_rise = glucose[5] - baseline
            if early_rise > thresholds['early_rise']:
                votes += 1
        
        # Rule 2: Peak rise
        if len(glucose) >= 12:
            peak_rise = np.max(glucose[:12]) - baseline
            if peak_rise > thresholds['peak_rise']:
                votes += 1
        
        # Rule 3: Sustained elevation
        if len(glucose) >= 18:
            sustained_rise = glucose[17] - baseline
            if sustained_rise > thresholds['sustained_rise']:
                votes += 1
        
        # Rule 4: Rate of rise
        if len(glucose) >= 6:
            rates = np.diff(glucose[:6])
            avg_rate = np.mean(rates)
            if avg_rate > thresholds['rate_rise']:
                votes += 1
        
        # Rule 5: Pattern consistency
        if len(glucose) >= 12:
            first_half = glucose[:6]
            second_half = glucose[6:12]
            if (np.max(second_half) > np.max(first_half) and 
                np.mean(second_half) > np.mean(first_half)):
                votes += 1
        
        # Rule 6: Gradual rise pattern
        if len(glucose) >= 8:
            rise_pattern = glucose[7] - glucose[0]
            if rise_pattern > thresholds['gradual_rise']:
                votes += 1
        
        # Rule 7: Mid-range elevation
        if len(glucose) >= 10:
            mid_rise = glucose[9] - baseline
            if mid_rise > thresholds['mid_rise']:
                votes += 1
        
        # Rule 8: Overall glucose range
        glucose_range = np.max(glucose) - np.min(glucose)
        if glucose_range > thresholds['glucose_range']:
            votes += 1
        
        # Rule 9: Positive slope dominance
        if len(glucose) >= 8:
            slopes = np.diff(glucose[:8])
            positive_slopes = np.sum(slopes > 0)
            if positive_slopes > len(slopes) * thresholds['slope_ratio']:
                votes += 1
        
        # Rule 10: Late rise pattern
        if len(glucose) >= 16:
            late_rise = glucose[15] - glucose[8]
            if late_rise > thresholds['late_rise']:
                votes += 1
        
        # Rule 11: Early acceleration
        if len(glucose) >= 4:
            early_accel = glucose[3] - glucose[0]
            if early_accel > thresholds['early_accel']:
                votes += 1
        
        # Rule 12: Steady rise pattern
        if len(glucose) >= 14:
            steady_rise = glucose[13] - glucose[6]
            if steady_rise > thresholds['steady_rise']:
                votes += 1
        
        # Rule 13: High baseline with rise
        if baseline > 140:
            if len(glucose) >= 6:
                if glucose[5] > baseline + thresholds['high_baseline']:
                    votes += 1
        
        # Rule 14: Multiple rise points
        if len(glucose) >= 12:
            rise_points = 0
            for i in range(1, 12):
                if glucose[i] > glucose[i-1]:
                    rise_points += 1
            if rise_points >= thresholds['rise_points']:
                votes += 1
        
        # Rule 15: Quick rise detection
        if len(glucose) >= 3:
            quick_rise = glucose[2] - glucose[0]
            if quick_rise > thresholds['quick_rise']:
                votes += 1
        
        # Rule 16: Peak timing
        if len(glucose) >= 12:
            peak_idx = np.argmax(glucose[:12])
            if 4 <= peak_idx <= 10:
                peak_height = glucose[peak_idx] - baseline
                if peak_height > thresholds['peak_height']:
                    votes += 1
        
        if votes >= voting_threshold:
            return 1
        return 0
    
    return detector

def evaluate_detector(detector, meal_windows, non_meal_windows):
    """
    Evaluate detector performance
    """
    # Test on meal windows
    meal_predictions = []
    for window in meal_windows:
        prediction = detector(window)
        meal_predictions.append(prediction)
    
    # Test on non-meal windows
    non_meal_predictions = []
    for window in non_meal_windows:
        prediction = detector(window)
        non_meal_predictions.append(prediction)
    
    # Calculate metrics
    y_true = [1] * len(meal_windows) + [0] * len(non_meal_windows)
    y_pred = meal_predictions + non_meal_predictions
    
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    return f1, accuracy, meal_predictions, non_meal_predictions

def load_data():
    """
    Load training data
    """
    print("Loading data...")
    insulin_df = pd.read_csv("InsulinData.csv")
    cgm_df = pd.read_csv("CGMData.csv")
    
    # Extract meal and non-meal windows (simplified version)
    # This is a placeholder - you'll need to implement the actual extraction
    meal_windows = [[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215]]  # Example meal window
    non_meal_windows = [
        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103],
        [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
    ]  # Example non-meal windows
    
    return meal_windows, non_meal_windows

def optimize_detector():
    """
    Automatically optimize detector until F1 >= 0.8
    """
    meal_windows, non_meal_windows = load_data()
    
    # Initial thresholds
    thresholds = {
        'early_rise': 8,
        'peak_rise': 15,
        'sustained_rise': 12,
        'rate_rise': 1.5,
        'gradual_rise': 12,
        'mid_rise': 10,
        'glucose_range': 25,
        'slope_ratio': 0.6,
        'late_rise': 8,
        'early_accel': 5,
        'steady_rise': 8,
        'high_baseline': 5,
        'rise_points': 7,
        'quick_rise': 4,
        'peak_height': 12
    }
    
    voting_threshold = 2
    best_f1 = 0
    best_accuracy = 0
    best_config = None
    iteration = 0
    
    print("Starting optimization...")
    print("Target: F1 Score >= 0.8")
    print("-" * 50)
    
    while best_f1 < 0.8 and iteration < 50:
        iteration += 1
        
        # Create detector with current thresholds
        detector = create_detector(thresholds, voting_threshold)
        
        # Evaluate detector
        f1, accuracy, meal_preds, non_meal_preds = evaluate_detector(detector, meal_windows, non_meal_windows)
        
        print(f"Iteration {iteration}:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Voting threshold: {voting_threshold}")
        print(f"  Meal predictions: {meal_preds}")
        print(f"  Non-meal predictions: {non_meal_preds}")
        print("-" * 30)
        
        # Update best if better
        if f1 > best_f1:
            best_f1 = f1
            best_accuracy = accuracy
            best_config = {
                'thresholds': thresholds.copy(),
                'voting_threshold': voting_threshold,
                'f1': f1,
                'accuracy': accuracy
            }
        
        # Adjust thresholds based on performance
        if f1 < 0.7:
            # Too conservative, lower thresholds
            for key in thresholds:
                if key != 'slope_ratio':
                    thresholds[key] = max(0, thresholds[key] - 1)
                else:
                    thresholds[key] = max(0.1, thresholds[key] - 0.1)
            voting_threshold = max(1, voting_threshold - 1)
        elif f1 < 0.75:
            # Slightly conservative, small adjustments
            for key in thresholds:
                if key != 'slope_ratio':
                    thresholds[key] = max(0, thresholds[key] - 0.5)
                else:
                    thresholds[key] = max(0.1, thresholds[key] - 0.05)
        elif f1 < 0.8:
            # Close to target, fine-tune
            for key in thresholds:
                if key != 'slope_ratio':
                    thresholds[key] = max(0, thresholds[key] - 0.2)
                else:
                    thresholds[key] = max(0.1, thresholds[key] - 0.02)
    
    print(f"\nOptimization completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    if best_f1 >= 0.8:
        print("Target achieved! F1 Score >= 0.8")
        return best_config
    else:
        print("Target not reached. Best configuration:")
        return best_config

if __name__ == "__main__":
    best_config = optimize_detector()
    print(f"\nBest configuration: {best_config}")