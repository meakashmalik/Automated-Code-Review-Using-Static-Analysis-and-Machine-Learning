import os
import json
import subprocess
import joblib
import pandas as pd # Ensure pandas is imported as we use it

# --- Configuration ---
# Paths to the saved models and vectorizer
MSG_ID_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'logistic_regression_pylint_classifier.joblib')
FP_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'false_positive_classifier.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'tfidf_vectorizer.joblib')

# Path to the Pylint executable (absolute path to avoid "command not found")
PYLINT_EXECUTABLE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'venv', 'bin', 'pylint')
# --- End Configuration ---


def load_models_and_vectorizer():
    """Loads both trained ML models and the TF-IDF vectorizer."""
    try:
        msg_id_model = joblib.load(MSG_ID_MODEL_PATH)
        fp_model = joblib.load(FP_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("All models and Vectorizer loaded successfully.")
        return msg_id_model, fp_model, vectorizer
    except FileNotFoundError as e:
        print(f"Error: Required model or vectorizer file not found: {e}. Make sure you've run 'model_trainer.py', 'false_positive_trainer.py', and 'feature_extractor.py'.")
        return None, None, None
    except Exception as e:
        print(f"Error loading models or vectorizer: {e}")
        return None, None, None

def run_pylint_on_text(code_text):
    """Runs Pylint on a given string of code and returns JSON output."""
    temp_file_path = "temp_code_for_pylint.py"
    with open(temp_file_path, "w") as f:
        f.write(code_text)

    try:
        result = subprocess.run(
            [PYLINT_EXECUTABLE_PATH, '--output-format=json', '--disable=C0114,C0115,C0116', temp_file_path],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 or result.returncode <= 31:
            return json.loads(result.stdout)
        else:
            print(f"Pylint encountered an internal error: {result.stderr}")
            return []
    except FileNotFoundError:
        print("Error: Pylint command not found at expected path. Ensure Pylint is installed in your venv.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from Pylint output for temp file. Raw output: {result.stdout[:500]}...")
        return []
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_code_snippet_from_text(code_text, line_number, context_lines=2):
    """Extracts code snippet from a string based on line number."""
    lines = code_text.splitlines()
    target_line_idx = line_number - 1

    line_content = lines[target_line_idx].strip() if 0 <= target_line_idx < len(lines) else ""
    context_before = [lines[i].strip() for i in range(max(0, target_line_idx - context_lines), target_line_idx)]
    context_after = [lines[i].strip() for i in range(target_line_idx + 1, min(len(lines), target_line_idx + 1 + context_lines))]

    return {
        "line_content": line_content,
        "context_before": context_before,
        "context_after": context_after
    }

def main():
    msg_id_model, fp_model, vectorizer = load_models_and_vectorizer()
    if msg_id_model is None or fp_model is None or vectorizer is None:
        return

    print("\n--- Enter Python Code to Review (Type 'END' on a new line to finish) ---")
    code_lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            code_lines.append(line)
        except EOFError: # Handles Ctrl+D/Ctrl+Z for end of input
            break

    new_code = "\n".join(code_lines)
    if not new_code.strip():
        print("No code entered. Exiting.")
        return

    print("\n--- Running Static Analysis (Pylint) ---")
    pylint_issues = run_pylint_on_text(new_code)

    if not pylint_issues:
        print("Pylint found no issues in the provided code, or encountered an error.")
        return

    print("\n--- Reviewing Pylint Issues with ML Models ---")
    review_results = []
    for issue in pylint_issues:
        line_number = issue.get('line', 1)
        snippet_data = get_code_snippet_from_text(new_code, line_number)

        combined_text = (snippet_data['line_content'].strip() + " " + # Added .strip() for consistency
                         issue.get('message', '').strip() + " " +
                         issue.get('symbol', '').strip()).strip()

        if not combined_text:
            continue

        new_features = vectorizer.transform([combined_text])

        # Predict original message ID
        predicted_message_id = msg_id_model.predict(new_features)[0]
        msg_id_confidence = msg_id_model.predict_proba(new_features).max()

        # Predict if it's a false positive
        is_false_positive_prediction = fp_model.predict(new_features)[0]
        fp_confidence = fp_model.predict_proba(new_features).max()

        # Generate a simple suggestion
        suggestion = ""
        if is_false_positive_prediction == 1:
            suggestion = "SUGGESTION: This might be a FALSE POSITIVE. Consider ignoring this warning or reviewing manually."
        else:
            suggestion = f"SUGGESTION: Address '{issue.get('symbol')}' (Type: {issue.get('type')}). Improve code by: {issue.get('message')}"

        review_results.append({
            "line": line_number,
            "code_snippet": snippet_data['line_content'],
            "original_pylint_symbol": issue.get('symbol'),
            "original_pylint_message": issue.get('message'),
            "predicted_message_id": predicted_message_id,
            "msg_id_confidence": msg_id_confidence,
            "is_false_positive_prediction": "YES" if is_false_positive_prediction == 1 else "NO",
            "fp_confidence": fp_confidence,
            "suggestion": suggestion
        })

    if not review_results:
        print("No issues processed by the ML models.")
        return

    print("\n--- Automated Code Review Results with ML Insights ---")
    for res in review_results:
        print(f"\n--- Issue on Line {res['line']} ---")
        print(f"  Code: '{res['code_snippet']}'")
        print(f"  Pylint Original: {res['original_pylint_symbol']} ({res['original_pylint_message']})")
        print(f"  ML Predicted Type: {res['predicted_message_id']} (Confidence: {res['msg_id_confidence']:.2f})")
        print(f"  ML Predicted False Positive: {res['is_false_positive_prediction']} (Confidence: {res['fp_confidence']:.2f})")
        print(f"  {res['suggestion']}")
        print("-" * 50)

if __name__ == "__main__":
    main()