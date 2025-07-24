import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Configuration ---
# Path to the processed issues data
PROCESSED_DATA_FILE = "/Users/manishsingh/Desktop/Mini_Project/automated-code-review/data/processed/all_pylint_issues_with_snippets.json" # Use the absolute path

# Path where the trained model will be saved
MODELS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')

# Path where the TF-IDF vectorizer is saved (from previous step)
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'tfidf_vectorizer.joblib')
# --- End Configuration ---

def load_issues_data(file_path):
    """Loads the JSON data containing all Pylint issues with snippets."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please ensure '{os.path.basename(file_path)}' exists in the 'data/processed/' directory.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check if the JSON is valid.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}")
        return None

def main():
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    print(f"Loading data from: {PROCESSED_DATA_FILE}")

    df_issues = load_issues_data(PROCESSED_DATA_FILE)

    if df_issues is None or df_issues.empty:
        print("No data to train the model. Exiting.")
        return

    # Combine relevant text for feature extraction (same as feature_extractor.py)
    df_issues['combined_text'] = df_issues['line_content'].fillna('') + " " + \
                                  df_issues['message'].fillna('') + " " + \
                                  df_issues['symbol'].fillna('')

    # Filter out entries where combined_text is empty or just whitespace
    df_issues = df_issues[df_issues['combined_text'].str.strip() != '']
    if df_issues.empty:
        print("No valid text data after filtering. Exiting.")
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

    # Transform the combined text using the loaded vectorizer
    X = vectorizer.transform(df_issues['combined_text'])
    print(f"Transformed text data into {X.shape[1]} features for {X.shape[0]} samples.")

    # Define our target variable (labels)
    # For this initial step, we are training to predict the Pylint message_id
    # This demonstrates a classification task.
    y = df_issues['message_id']
    print(f"Unique message IDs (labels) found: {y.nunique()}")

    # Split data into training and testing sets
    # test_size=0.2 means 20% of the data will be used for testing, 80% for training
    # random_state ensures results are reproducible (same split every time)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # Initialize and train the Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear') # max_iter for convergence, solver for efficiency
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}") # Proportion of correctly predicted issues

    # A detailed report showing precision, recall, f1-score for each message_id
    print("\nClassification Report:")
    # Handle cases where some labels might not be in the test set after stratification if very few samples
    print(classification_report(y_test, y_pred, zero_division=0)) 

    # Save the trained model
    model_filename = "logistic_regression_pylint_classifier.joblib"
    model_path = os.path.join(MODELS_OUTPUT_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"\nTrained model saved to: {model_path}")

if __name__ == "__main__":
    main()