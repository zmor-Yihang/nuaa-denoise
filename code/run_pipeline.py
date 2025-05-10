import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)
    print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    scripts = [
        "vmd_saveImf.py",
        "imf_feature_extraction.py",
        "imf_clustering.py",
        "signal_reconstruction.py"
    ]
    
    for script in scripts:
        run_script(script)
