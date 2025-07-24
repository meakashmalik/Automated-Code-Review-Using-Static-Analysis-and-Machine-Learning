import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# --- Configuration ---
# Path to the processed issues data (main dataset)
PROCESSED_DATA_FILE = "/Users/manishsingh/Desktop/Mini_Project/automated-code-review/data/processed/all_pylint_issues_with_snippets.json" # Your absolute path

# Path to your manually labeled data
# Path to your manually labeled data (temporarily hardcoded for troubleshooting)
MANUAL_LABELS_FILE = "/Users/manishsingh/Desktop/Mini_Project/automated-code-review/data/processed/manual_labels.csv"

# Path where the trained model will be saved
MODELS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')

# Path where the TF-IDF vectorizer is saved (from Step 8)
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'tfidf_vectorizer.joblib')
# --- End Configuration ---

def load_data(file_path):
    """Loads JSON/CSV data into a Pandas DataFrame."""
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}")
        return None

def main():
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    print(f"Loading main issues data from: {PROCESSED_DATA_FILE}")
    df_issues = load_data(PROCESSED_DATA_FILE)

    if df_issues is None or df_issues.empty:
        print("No main issues data to process. Exiting.")
        return

    print(f"Loading manual labels from: {MANUAL_LABELS_FILE}")
    df_labels = load_data(MANUAL_LABELS_FILE)

    if df_labels is None or df_labels.empty:
        print("No manual labels found. Cannot train false positive model. Exiting.")
        return

    # Prepare 'combined_text' for feature extraction (same as before)
    df_issues['combined_text'] = df_issues['line_content'].fillna('') + " " + \
                                  df_issues['message'].fillna('') + " " + \
                                  df_issues['symbol'].fillna('')
    df_issues = df_issues[df_issues['combined_text'].str.strip() != '']

    # Create a unique key for joining/matching issues
    # We'll use message_id, line_content, and line number for robust matching
    df_issues['unique_key'] = df_issues['message_id'] + '|' + \
                              df_issues['line_content'].fillna('') + '|' + \
                              df_issues['line'].astype(str) # Convert line number to string for key

    df_labels['unique_key'] = df_labels['message_id'] + '|' + \
                              df_labels['code_snippet'].fillna('') + '|' + \
                              'UNKNOWN_LINE' # Line number is not in manual_labels.csv

    # Initialize 'is_false_positive' column to 0 (True Positive by default)
    df_issues['is_false_positive'] = 0

    # Update labels based on manual_labels.csv
    # For demonstration, we'll prioritize applying false positive labels
    # to *any* matching message_id, assuming manual labels represent types of FPs.
    print(f"Applying {len(df_labels)} manual labels based on message_id...")

    # Create a dictionary of message_id -> is_false_positive from manual labels
    # If a message_id appears multiple times in manual_labels, the last one will win.
    fp_label_map = {}
    for index, row in df_labels.iterrows():
        fp_label_map[row['message_id']] = row['is_false_positive']

    # Apply labels to the main dataframe
    # If a message_id from the main data matches one in our manual labels,
    # we'll assign the corresponding false positive label.
    # Issues not in manual_labels or explicitly marked as FP will remain 0.

    df_issues['is_false_positive'] = df_issues['message_id'].apply(lambda x: fp_label_map.get(x, 0))

    # Count how many issues now have the '1' (False Positive) label
    changed_to_fp_count = df_issues['is_false_positive'].sum()
    print(f"Set {changed_to_fp_count} issues to '1' (False Positive) based on message_id lookup.")
    print(f"Distribution of labels: \n{df_issues['is_false_positive'].value_counts()}")


    # Filter out samples with no valid features (combined_text might be empty for some module-level issues)
    df_issues_filtered = df_issues[df_issues['combined_text'].str.strip() != '']
    if df_issues_filtered.empty:
        print("No valid data after filtering for combined_text. Exiting.")
        return

    # Load the pre-trained TF-IDF vectorizer
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        print(f"Loaded TF-IDF vectorizer from: {VECTORIZER_PATH}")
    except FileNotFoundError:
        print(f"Error: TF-IDF vectorizer not found at {VECTORIZER_PATH}. Please run feature_extractor.py first.")
        return
    except Exception as e:
        print(f"Error loading TF-IDF vectorizer: {e}")
        return

    # Transform the combined text into numerical features
    X = vectorizer.transform(df_issues_filtered['combined_text'])
    print(f"Transformed text data into {X.shape[1]} features for {X.shape[0]} samples.")

    # Our new target variable (label) for False Positive detection
    y = df_issues_filtered['is_false_positive']

    # Check for sufficient number of samples in both classes (0 and 1)
    # If one class has very few, train_test_split with stratify might fail or training is not meaningful
    if y.nunique() < 2 or y.value_counts().min() < 2:
        print("Warning: Not enough diverse labels (true/false positive) to train a robust model.")
        print("Please ensure your 'manual_labels.csv' has at least 2 True Positives (0) and 2 False Positives (1).")
        print("Proceeding without stratification or with limited training due to sparse labels.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        # Stratify only if enough samples in each class
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # Initialize and train the Logistic Regression model for binary classification
    print("Training Logistic Regression model for False Positive detection...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear', class_weight='balanced') # 'balanced' helps with uneven class counts
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("\n--- False Positive Model Evaluation ---")
    y_pred = model.predict(X_test)

    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['True Positive (0)', 'False Positive (1)'], zero_division=0))

    # Specific metrics for False Positives (class 1)
    # Precision: Of all predicted FPs, how many were actually FPs?
    # Recall: Of all actual FPs, how many did we correctly identify?
    print(f"Precision for False Positives (class 1): {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.2f}")
    print(f"Recall for False Positives (class 1): {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.2f}")
    print(f"F1-Score for False Positives (class 1): {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.2f}")

    # Save the trained false positive model
    model_filename = "false_positive_classifier.joblib"
    model_path = os.path.join(MODELS_OUTPUT_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"\nTrained False Positive model saved to: {model_path}")

if __name__ == "__main__":
    main()