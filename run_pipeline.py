import subprocess
import sys

def run_script(script_name):
    """Runs a Python script using the same interpreter and captures its output."""
    print(f"\n--- Running {script_name} ---")
    try:
        # Use sys.executable to ensure we're using the python from the current virtual env
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,        # This will raise an exception if the script fails
            capture_output=True, # Capture stdout and stderr
            text=True          # Decode output as text
        )
        print(result.stdout)
        if result.stderr:
            print("--- Errors/Warnings ---")
            print(result.stderr)
        print(f"--- {script_name} finished successfully ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR running {script_name} !!!")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"!!! ERROR: Script '{script_name}' not found. !!!")
        return False

if __name__ == "__main__":
    print("🚀 Starting the F1 Predictor update pipeline...")

    # Step 1: Run the data collection script
    if run_script("data_collection.py"):
        # Step 2: If data collection is successful, run the model training script
        run_script("model_training.py")
    
    print("\n✅ Pipeline finished.")

