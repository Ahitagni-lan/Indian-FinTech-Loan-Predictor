"""
Indian FinTech Loan Risk Predictor (ML)

This script builds an end-to-end ML model for a classic FinTech problem,
specifically tailored to the Indian market.

It is 100% self-contained and requires no external data files or APIs.

It will:
1. Generate a synthetic dataset of 2,000 loan applicants with
   India-specific features (CIBIL, KYC, UPI, City Tier, etc.).
2. Train a Random Forest model to classify applicants as "Repay" (0) or "Default" (1).
3. Print a full evaluation report (Accuracy, Precision, Recall).
4. Plot a bar chart showing which features were most important for the model's decision.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def create_synthetic_dataset():
    """
    Generates a synthetic dataset of loan applicants,
    with features relevant to the Indian FinTech context.
    """
    print("Generating synthetic Indian loan applicant dataset...")
    
    # We will generate 10 features in total.
    # 8 will be informative, 1 redundant, 1 noise.
    # This simulates a real-world scenario where some data is useful,
    # some is correlated, and some is just noise.
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        n_repeated=0, # n_features - n_informative - n_redundant = 1 noise feature
        n_clusters_per_class=1,
        weights=[0.85, 0.15], # 85% repay, 15% default (realistic imbalance)
        class_sep=0.8,
        random_state=42
    )
    
    # Give the features realistic, India-specific FinTech names
    feature_names = [
        'CIBIL_Score',           # Informative
        'Annual_Income_INR',     # Informative
        'Loan_Amount_INR',       # Informative
        'Debt_to_Income_Ratio',  # Informative
        'Employment_Years',      # Informative
        'Is_KYC_Verified',       # Informative (Aadhaar/PAN)
        'Avg_Monthly_UPI_Volume',# Informative
        'City_Tier',             # Informative (1, 2, or 3)
        'Loan_History_Defaults', # Redundant (Correlated with CIBIL)
        'Noise_Feature'          # Noise (Should have low importance)
    ]
    
    # Convert to a pandas DataFrame for easier handling
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y, name="Default_Status")
    
    print(f"Dataset generated with {len(X_df)} samples.")
    print("Sample Data:\n", X_df.head())
    print("\nTarget Distribution (0=Repay, 1=Default):\n", y_s.value_counts(normalize=True))
    
    return X_df, y_s, feature_names

def train_ml_model(X, y):
    """
    Splits the data, trains a Random Forest, and returns the model.
    """
    print("\nSplitting data and training Random Forest model...")
    
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize the Random Forest Classifier
    # class_weight='balanced' is crucial for imbalanced data,
    # as it tells the model to pay more attention to the rare "Default" class.
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training complete.")
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints key performance metrics.
    """
    print("\n" + "="*30)
    print("Model Evaluation Results")
    print("="*30)
    
    y_pred = model.predict(X_test)
    
    # --- Accuracy ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\n--- Confusion Matrix ---")
    print("(Rows = Actual, Cols = Predicted)")
    print(confusion_matrix(y_test, y_pred))
    
    # --- Classification Report ---
    # Precision (1): Of all the times we *predicted* "Default", how many were correct?
    # Recall (1): Of all the *actual* "Default" cases, how many did we catch?
    # In fraud/risk, Recall is often the most important metric.
    print("\n--- Classification Report ---")
    print(classification_record := classification_report(y_test, y_pred, target_names=["Repay (0)", "Default (1)"]))

    # Check if the report string exists before trying to parse
    if classification_record:
        try:
            # Extract recall for the "Default" class
            report_lines = classification_record.split('\n')
            default_line = [line for line in report_lines if "Default (1)" in line][0]
            default_stats = default_line.split()
            recall_val = float(default_stats[3])
            print(f"\nKey Takeaway: The model successfully identified {recall_val:.2%} of all *actual* defaults.")
        except (IndexError, ValueError) as e:
            print(f"\nCould not parse classification report for key takeaway. Error: {e}")
    else:
        print("\nClassification report was empty, skipping key takeaway.")

    print("="*30)

def plot_feature_importance(model, feature_names):
    """
    Gets feature importances from the model and plots them.
    """
    print("\nGenerating feature importance plot...")
    
    importances = model.feature_importances_
    
    # Create a pandas Series for easy sorting and plotting
    feature_importance_series = pd.Series(importances, index=feature_names)
    
    # Sort from most to least important
    sorted_importances = feature_importance_series.sort_values(ascending=False)
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    sorted_importances.plot(kind='bar', color='deepskyblue')
    plt.title('Feature Importance for Indian Loan Default Prediction')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Adjust layout to make room for labels
    
    print("Displaying plot. Close the plot window to exit the script.")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Create the data
    X_data, y_data, features = create_synthetic_dataset()
    
    # 2. Train the model
    # We pass .values to avoid a warning about feature names in some sklearn versions
    trained_model, X_test_data, y_test_data = train_ml_model(X_data.values, y_data.values)
    
    # 3. Evaluate its performance
    evaluate_model(trained_model, X_test_data, y_test_data)
    
    # 4. Show which features mattered
    plot_feature_importance(trained_model, features)
