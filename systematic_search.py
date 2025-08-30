import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score

# Import the existing functions
from train import extract_meal_24, extract_nomeal_24

def test_meal_detector_variant(variant_params):
    """
    Test a specific variant of the meal detector
    """
    def meal_detector_variant(glucose_data):
        if len(glucose_data) < 24:
            return 0
        
        glucose = np.array(glucose_data)
        baseline = np.mean(glucose[:3])
        
        # Extract features based on variant parameters
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
        
        # Scoring based on variant parameters
        score = 0.0
        
        # Early response scoring
        if early_rise > variant_params['early_rise_threshold']:
            score += variant_params['early_rise_weight']
        
        if early_slope > variant_params['early_slope_threshold']:
            score += variant_params['early_slope_weight']
        
        # Mid response scoring
        if mid_rise > variant_params['mid_rise_threshold']:
            score += variant_params['mid_rise_weight']
        
        # Pattern scoring
        if total_range > variant_params['range_threshold']:
            score += variant_params['range_weight']
        
        if positive_rate_ratio > variant_params['rate_ratio_threshold']:
            score += variant_params['rate_ratio_weight']
        
        # Decision threshold
        if score >= variant_params['decision_threshold']:
            return 1
        
        return 0
    
    return meal_detector_variant

def evaluate_variant(detector_func, meal_windows, nomeal_windows):
    """
    Evaluate a meal detector variant
    """
    meal_predictions = []
    for window in meal_windows:
        pred = detector_func(window)
        meal_predictions.append(pred)
    
    nomeal_predictions = []
    for window in nomeal_windows:
        pred = detector_func(window)
        nomeal_predictions.append(pred)
    
    # Create true labels
    y_true = [1] * len(meal_windows) + [0] * len(nomeal_windows)
    y_pred = meal_predictions + nomeal_predictions
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    meal_accuracy = np.mean(meal_predictions)
    nomeal_accuracy = 1 - np.mean(nomeal_predictions)
    
    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'meal_accuracy': meal_accuracy,
        'nomeal_accuracy': nomeal_accuracy
    }

def systematic_search():
    """
    Perform systematic search for the best meal detector
    """
    print("Loading data...")
    insulin_df = pd.read_csv("InsulinData.csv")
    cgm_df = pd.read_csv("CGMData.csv")
    
    print("Extracting windows...")
    meal_windows = extract_meal_24(insulin_df, cgm_df)
    nomeal_windows = extract_nomeal_24(insulin_df, cgm_df)
    
    print(f"Found {len(meal_windows)} meal windows and {len(nomeal_windows)} non-meal windows")
    
    # Define parameter ranges for systematic search
    param_ranges = {
        'early_rise_threshold': [0, 1, 2, 3, 4, 5, 6, 8, 10, 12],
        'early_slope_threshold': [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        'mid_rise_threshold': [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20],
        'range_threshold': [0, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30],
        'rate_ratio_threshold': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'early_rise_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        'early_slope_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'mid_rise_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        'range_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'rate_ratio_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
        'decision_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
    }
    
    best_result = None
    best_f1 = 0.0
    best_params = None
    
    print("Starting systematic search...")
    
    # Try different combinations systematically
    attempt_count = 0
    max_attempts = 300
    
    # Focus on promising parameter ranges
    for early_rise_thresh in [0, 1, 2, 3, 4, 5]:
        for early_slope_thresh in [0, 0.1, 0.2, 0.3, 0.5]:
            for mid_rise_thresh in [0, 1, 2, 3, 4, 5, 6, 8]:
                for range_thresh in [0, 2, 3, 5, 8, 10, 12]:
                    for rate_ratio_thresh in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                        for early_rise_weight in [0.2, 0.3, 0.4, 0.5, 0.6]:
                            for mid_rise_weight in [0.2, 0.3, 0.4, 0.5, 0.6]:
                                for decision_thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
                                    attempt_count += 1
                                    if attempt_count > max_attempts:
                                        break
                                    
                                    # Create variant parameters
                                    variant_params = {
                                        'early_rise_threshold': early_rise_thresh,
                                        'early_slope_threshold': early_slope_thresh,
                                        'mid_rise_threshold': mid_rise_thresh,
                                        'range_threshold': range_thresh,
                                        'rate_ratio_threshold': rate_ratio_thresh,
                                        'early_rise_weight': early_rise_weight,
                                        'early_slope_weight': 0.3,
                                        'mid_rise_weight': mid_rise_weight,
                                        'range_weight': 0.3,
                                        'rate_ratio_weight': 0.2,
                                        'decision_threshold': decision_thresh
                                    }
                                    
                                    # Test this variant
                                    detector_func = test_meal_detector_variant(variant_params)
                                    result = evaluate_variant(detector_func, meal_windows, nomeal_windows)
                                    
                                    if result['f1_score'] > best_f1:
                                        best_f1 = result['f1_score']
                                        best_result = result
                                        best_params = variant_params
                                        print(f"Attempt {attempt_count}: New best F1 = {best_f1:.4f}")
                                        print(f"  Params: {best_params}")
                                        print(f"  Result: {result}")
                                    
                                    if attempt_count % 50 == 0:
                                        print(f"Completed {attempt_count} attempts. Best F1 so far: {best_f1:.4f}")
                                
                                if attempt_count > max_attempts:
                                    break
                            if attempt_count > max_attempts:
                                break
                        if attempt_count > max_attempts:
                            break
                    if attempt_count > max_attempts:
                        break
                if attempt_count > max_attempts:
                    break
            if attempt_count > max_attempts:
                break
        if attempt_count > max_attempts:
            break
    
    print(f"\nSearch completed! Best result from {attempt_count} attempts:")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Result: {best_result}")
    
    return best_params, best_result

if __name__ == "__main__":
    best_params, best_result = systematic_search()