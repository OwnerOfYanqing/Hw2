import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from train import optimized_meal_detector

def evaluate_performance():
    """
    Evaluate the performance of the optimized meal detector
    """
    print("Loading data...")
    insulin_df = pd.read_csv("InsulinData.csv")
    cgm_df = pd.read_csv("CGMData.csv")
    
    # Extract meal windows
    print("Extracting meal windows...")
    meal_windows = []
    
    for _, row in insulin_df.iterrows():
        if pd.notna(row['BWZ Carb Input (grams)']) and row['BWZ Carb Input (grams)'] > 0:
            # Find the corresponding CGM data
            meal_time = row['Date']
            meal_idx = cgm_df[cgm_df['Date'] == meal_time].index
            
            if len(meal_idx) > 0:
                start_idx = meal_idx[0]
                if start_idx + 24 <= len(cgm_df):
                    glucose_window = cgm_df.iloc[start_idx:start_idx+24]['Sensor Glucose (mg/dL)'].values
                    if len(glucose_window) == 24 and not np.any(np.isnan(glucose_window)):
                        meal_windows.append(glucose_window)
    
    print(f"Found {len(meal_windows)} meal windows")
    
    # Extract non-meal windows
    print("Extracting non-meal windows...")
    non_meal_windows = []
    
    # Sample some non-meal periods
    for i in range(0, len(cgm_df) - 24, 100):  # Sample every 100th window
        glucose_window = cgm_df.iloc[i:i+24]['Sensor Glucose (mg/dL)'].values
        if len(glucose_window) == 24 and not np.any(np.isnan(glucose_window)):
            non_meal_windows.append(glucose_window)
    
    print(f"Found {len(non_meal_windows)} non-meal windows")
    
    # Test the detector
    print("Testing optimized meal detector...")
    
    # Test on meal windows
    meal_predictions = []
    for window in meal_windows:
        prediction = optimized_meal_detector(window)
        meal_predictions.append(prediction)
    
    # Test on non-meal windows
    non_meal_predictions = []
    for window in non_meal_windows:
        prediction = optimized_meal_detector(window)
        non_meal_predictions.append(prediction)
    
    # Calculate metrics
    y_true = [1] * len(meal_windows) + [0] * len(non_meal_windows)
    y_pred = meal_predictions + non_meal_predictions
    
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    meal_detection_accuracy = np.mean(meal_predictions)
    non_meal_detection_accuracy = 1 - np.mean(non_meal_predictions)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Meal detection accuracy: {meal_detection_accuracy:.4f}")
    print(f"Non-meal detection accuracy: {non_meal_detection_accuracy:.4f}")
    
    return f1, accuracy, meal_detection_accuracy, non_meal_detection_accuracy

if __name__ == "__main__":
    f1, accuracy, meal_acc, non_meal_acc = evaluate_performance()
    print(f"\nFinal Results:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Meal detection accuracy: {meal_acc:.4f}")
    print(f"Non-meal detection accuracy: {non_meal_acc:.4f}")