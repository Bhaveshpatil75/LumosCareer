import os
import sys
import subprocess

def run_script(script_name):
    print(f"\n--- Running {script_name} ---")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {script_name} not found in the current directory.")
        sys.exit(1)

def main():
    print("Starting database seed process...")
    
    # Run migrations first to ensure tables exist
    print("\n--- Running migrations ---")
    try:
        subprocess.run([sys.executable, "manage.py", "migrate"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running migrations: {e}")
        sys.exit(1)
    
    # List of scripts to run in logical order
    scripts_to_run = [
        "populate_db.py",
        "load_initial_data.py",
        "seed_db.py"
    ]
    
    for script in scripts_to_run:
        # Check if file exists before running
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Skipping {script} (file not found)")
            
    print("\n✅ Database seeding complete!")

if __name__ == "__main__":
    main()
