import sys
import os
from flask import Flask, render_template, request

# Define path_to_add BEFORE using it
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'ml_model'))
sys.path.append(path_to_add)

print(f"Attempting to add to sys.path: {path_to_add}") # <-- THIS LINE WILL NOW WORK

# Now you can import your functions from prediction_service.py
# ...
from prediction_service import run_pylint_on_text, load_models_and_vectorizer, get_code_snippet_from_text

app = Flask(__name__)

# Load models once when the app starts
msg_id_model, fp_model, vectorizer = load_models_and_vectorizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    review_results = None
    if request.method == 'POST':
        user_code = request.form['code_input']
        if user_code:
            # This is where you integrate your prediction_service logic
            # You'll need to make run_pylint_on_text and other functions
            # callable directly and adapt the results.

            pylint_issues = run_pylint_on_text(user_code) # Call your existing function

            if pylint_issues and msg_id_model and fp_model and vectorizer:
                review_results = []
                for issue in pylint_issues:
                    line_number = issue.get('line', 1)
                    snippet_data = get_code_snippet_from_text(user_code, line_number)

                    combined_text = (snippet_data['line_content'].strip() + " " +
                                     issue.get('message', '').strip() + " " +
                                     issue.get('symbol', '').strip()).strip()

                    if not combined_text:
                        continue

                    new_features = vectorizer.transform([combined_text])

                    predicted_message_id = msg_id_model.predict(new_features)[0]
                    msg_id_confidence = msg_id_model.predict_proba(new_features).max()

                    is_false_positive_prediction = fp_model.predict(new_features)[0]
                    fp_confidence = fp_model.predict_proba(new_features).max()

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
                        "msg_id_confidence": f"{msg_id_confidence:.2f}",
                        "is_false_positive_prediction": "YES" if is_false_positive_prediction == 1 else "NO",
                        "fp_confidence": f"{fp_confidence:.2f}",
                        "suggestion": suggestion
                    })
            else:
                review_results = [{"error": "Could not process code or models not loaded."}]
    return render_template('index.html', review_results=review_results)

if __name__ == '__main__':
    app.run(debug=True) # debug=True allows automatic reload on code changes