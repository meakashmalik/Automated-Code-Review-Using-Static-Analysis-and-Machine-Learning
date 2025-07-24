import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib # To save our TF-IDF model

# --- Configuration ---
# Path to the processed issues data
# Path to the processed issues data (temporarily hardcoded for troubleshooting)
PROCESSED_DATA_FILE = "/Users/manishsingh/Desktop/Mini_Project/automated-code-review/data/processed/all_pylint_issues_with_snippets.json"

# Path to save the extracted features and the TF-IDF vectorizer model
FEATURES_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'features')
# --- End Configuration ---

def load_issues_data(file_path):
    """Loads the JSON data containing all Pylint issues with snippets."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}")
        return None

def create_text_features(df):
    """
    Combines relevant text fields and applies TF-IDF vectorization.
    """
    if df is None or df.empty:
        print("No data to process for feature creation.")
        return None, None

    # Combine the code line, message, and symbol into one text string for TF-IDF
    # This creates a comprehensive text representation for each issue
    df['combined_text'] = df['line_content'].fillna('') + " " + \
                          df['message'].fillna('') + " " + \
                          df['symbol'].fillna('')

    # Initialize the TF-IDF Vectorizer
    # max_features limits the number of unique words (features) to the most important ones.
    # stop_words='english' removes common English words that don't add much meaning.
    # We can expand on this for code-specific stop words later.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    print("Fitting TF-IDF Vectorizer and transforming text data...")
    # Fit the vectorizer to our combined text and transform the text into numerical features
    text_features = vectorizer.fit_transform(df['combined_text'])

    print(f"Created {text_features.shape[1]} text features for {text_features.shape[0]} issues.")
    return text_features, vectorizer

def main():
    os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)
    print(f"Loading data from: {PROCESSED_DATA_FILE}")

    df_issues = load_issues_data(PROCESSED_DATA_FILE)

    if df_issues is not None:
        # For our initial ML experiment, let's create a placeholder 'label' column.
        # In a real scenario, this would be based on human feedback (e.g., true positive/false positive).
        # For now, let's just use the message_id as a dummy label for demonstration,
        # if we were to try to "predict" the type of issue.
        # Or, if we're aiming for false positive reduction, this column would later
        # be populated by "is_false_positive" (0 or 1).
        print("Adding a dummy 'label' column (message_id for now).")
        df_issues['label'] = df_issues['message_id'] # We will change this later for true/false positive labeling

        # Create text features (numerical representation of code + message)
        features_sparse_matrix, vectorizer = create_text_features(df_issues)

        if features_sparse_matrix is not None:
            # Save the processed features (as a compressed numpy array or similar if it gets very large)
            # For now, let's save the DataFrame with a placeholder for the text features,
            # and the vectorizer itself to transform new unseen data later.

            # We can't directly save the sparse matrix into CSV easily,
            # but we can save the vectorizer and the original DataFrame with other columns.
            print(f"Saving TF-IDF vectorizer to {os.path.join(FEATURES_OUTPUT_DIR, 'tfidf_vectorizer.joblib')}")
            joblib.dump(vectorizer, os.path.join(FEATURES_OUTPUT_DIR, 'tfidf_vectorizer.joblib'))

            # To save the features themselves, we often convert to a dense array if size allows,
            # or save as a sparse format (like .npz). For now, let's just confirm creation.
            # We will re-generate these features when training.
            print("TF-IDF features created. The vectorizer has been saved.")
            print("Next, we will use these features to train a simple model.")
    else:
        print("Skipping feature extraction due to missing data.")


if __name__ == "__main__":
    main()