import os
import json

# --- Configuration ---
# Path to the directory where Pylint reports are saved
# This points to 'automated-code-review/data/reports/'
REPORTS_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'reports')

# Path to the directory where raw code is stored (to read code snippets)
# This points to 'automated-code-review/data/raw/'
CODE_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Path where the processed data will be saved
# This will create 'automated-code-review/data/processed/issues.json'
PROCESSED_DATA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
# --- End Configuration ---

def get_code_snippet(file_path, line_number, context_lines=2):
    """
    Reads a file and returns the specified line plus context lines around it.
    Returns a dictionary with 'line_content', 'line_number', 'context_before', 'context_after'.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Adjust to 0-based index for list access
        target_line_idx = line_number - 1

        line_content = lines[target_line_idx].strip() if 0 <= target_line_idx < len(lines) else ""

        context_before = [
            lines[i].strip() for i in range(max(0, target_line_idx - context_lines), target_line_idx)
        ]
        context_after = [
            lines[i].strip() for i in range(target_line_idx + 1, min(len(lines), target_line_idx + 1 + context_lines))
        ]

        return {
            "line_content": line_content,
            "line_number": line_number,
            "context_before": context_before,
            "context_after": context_after
        }
    except FileNotFoundError:
        print(f"  Code file not found: {file_path}")
        return None
    except IndexError:
        print(f"  Line number {line_number} out of range in {file_path}")
        return None
    except Exception as e:
        print(f"  Error reading file {file_path}: {e}")
        return None

def main():
    os.makedirs(PROCESSED_DATA_OUTPUT_DIR, exist_ok=True)
    all_issues_data = []

    print(f"Processing reports from: {REPORTS_ROOT_DIR}")
    print(f"Reading code from: {CODE_ROOT_DIR}")
    print(f"Saving processed data to: {PROCESSED_DATA_OUTPUT_DIR}")

    # Walk through the reports directory to find all JSON report files
    for root, _, files in os.walk(REPORTS_ROOT_DIR):
        for file in files:
            if file.endswith('_pylint.json'):
                report_filepath = os.path.join(root, file)
                # Determine the original Python file path based on the report path
                # Example: data/reports/python-mini-projects/project_name/file_name_pylint.json
                # Should map to: data/raw/python-mini-projects/project_name/file_name.py

                # Remove '_pylint.json' and add '.py'
                relative_report_path = os.path.relpath(report_filepath, REPORTS_ROOT_DIR)
                original_py_file_relative_path = relative_report_path.replace('_pylint.json', '.py')

                # Construct the full path to the original code file
                original_py_file_path = os.path.join(CODE_ROOT_DIR, original_py_file_relative_path)

                if not os.path.exists(original_py_file_path):
                    print(f"Skipping report: Corresponding code file not found for {original_py_file_path}")
                    continue

                print(f"  Processing report: {relative_report_path}")

                try:
                    with open(report_filepath, 'r') as f:
                        pylint_report = json.load(f)

                    for issue in pylint_report:
                        issue_data = {
                            "file_path": original_py_file_relative_path,
                            "line": issue.get('line'),
                            "column": issue.get('column'),
                            "message_id": issue.get('message-id'),
                            "message_type": issue.get('type'),
                            "symbol": issue.get('symbol'),
                            "message": issue.get('message'),
                            "module": issue.get('module')
                        }

                        # Get the actual code snippet
                        if issue_data['line']:
                            snippet = get_code_snippet(original_py_file_path, issue_data['line'])
                            if snippet:
                                issue_data.update(snippet) # Add snippet details to the issue data
                        else:
                            # For module-level messages that don't have a specific line
                            issue_data['line_content'] = ""
                            issue_data['context_before'] = []
                            issue_data['context_after'] = []

                        all_issues_data.append(issue_data)

                except json.JSONDecodeError:
                    print(f"    Error: Could not decode JSON from {report_filepath}")
                except Exception as e:
                    print(f"    An unexpected error occurred processing {report_filepath}: {e}")

    print(f"\n--- Processing Complete ---")
    print(f"Total issues extracted: {len(all_issues_data)}")

    # Save all extracted issues to a single JSON file
    output_file = os.path.join(PROCESSED_DATA_OUTPUT_DIR, 'all_pylint_issues_with_snippets.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_issues_data, f, indent=4, ensure_ascii=False)
    print(f"All processed issues saved to: {output_file}")

if __name__ == "__main__":
    main()