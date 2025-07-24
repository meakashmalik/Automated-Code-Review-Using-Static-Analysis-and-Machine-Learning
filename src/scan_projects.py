import os
import subprocess
import json # To save the Pylint output in a structured way

# --- Configuration ---
# Path to the directory where you cloned the repositories
# Make sure this path is correct for your setup!
# It should point to 'automated-code-review/data/raw/'
PROJECTS_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Path where you want to save the Pylint reports
# This will create 'automated-code-review/data/reports/'
REPORTS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'reports')
# --- End Configuration ---

def run_pylint_on_file(file_path):
    """Runs Pylint on a single Python file and returns the JSON output."""
    print(f"  Running Pylint on: {file_path}")
    try:
        # --output-format=json tells Pylint to give us output in JSON format, which is easy for computers to read.
        # --disable=C0114,C0115,C0116 disables "missing docstring" warnings for now, as many small examples don't have them.
        # This makes the output cleaner for our initial focus on other issues.
        result = subprocess.run(
            ['pylint', '--output-format=json', '--disable=C0114,C0115,C0116', file_path],
            capture_output=True, # Capture the output
            text=True,           # Output as text (string)
            check=False          # Don't raise an error if Pylint finds issues (we expect it to)
        )
        if result.returncode == 0 or result.returncode <= 31: # Pylint returns codes for different issues
            return json.loads(result.stdout)
        else:
            print(f"    Pylint error running on {file_path}: {result.stderr}")
            return None
    except FileNotFoundError:
        print("    Error: Pylint command not found. Make sure Pylint is installed and your virtual environment is active.")
        return None
    except json.JSONDecodeError:
        print(f"    Error decoding JSON from Pylint output for {file_path}: {result.stdout[:200]}...") # Show beginning of output if not JSON
        return None

def main():
    # Create the reports output directory if it doesn't exist
    os.makedirs(REPORTS_OUTPUT_DIR, exist_ok=True)
    print(f"Scanning projects in: {PROJECTS_ROOT_DIR}")
    print(f"Saving reports to: {REPORTS_OUTPUT_DIR}")

    all_reports = {} # Dictionary to store all reports

    # Walk through the root directory to find all .py files
    for root, _, files in os.walk(PROJECTS_ROOT_DIR):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # We only want to process files from the specific mini-projects folder
                if 'python-mini-projects' in file_path:
                    relative_path = os.path.relpath(file_path, PROJECTS_ROOT_DIR)
                    print(f"Processing: {relative_path}")
                    pylint_output = run_pylint_on_file(file_path)

                    if pylint_output:
                        # Store the report, using the relative path as key
                        # This will allow us to easily find the report for a given file later
                        all_reports[relative_path] = pylint_output
                        # Optionally, save each report individually as well (good for debugging)
                        report_filename = os.path.basename(file_path).replace('.py', '_pylint.json')
                        # Ensure subdirectories match the input structure in reports folder
                        report_sub_dir = os.path.join(REPORTS_OUTPUT_DIR, os.path.dirname(relative_path))
                        os.makedirs(report_sub_dir, exist_ok=True)
                        report_filepath = os.path.join(report_sub_dir, report_filename)
                        with open(report_filepath, 'w') as f:
                            json.dump(pylint_output, f, indent=4)
                    else:
                        print(f"  No Pylint output for {relative_path}")

    print("\n--- Scan Complete ---")
    print(f"Total files processed: {len(all_reports)}")
    # Optionally, you could save the combined all_reports dictionary to one big JSON file here too.

if __name__ == "__main__":
    main()