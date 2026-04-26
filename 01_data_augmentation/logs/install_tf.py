import sys
import subprocess

print(f"Using python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    print("Attempting to install tensorflow...")
    # Add --no-cache-dir to avoid some memory issues, and -v for verbose output
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "tensorflow", "--no-cache-dir"],
        capture_output=True, text=True
    )
    print("Return code:", result.returncode)
    
    with open("logs/pip_tf_stdout.log", "w") as f:
        f.write(result.stdout)
    with open("logs/pip_tf_stderr.log", "w") as f:
        f.write(result.stderr)
        
    print("Pip stdout/stderr written to logs/pip_tf_stdout.log and logs/pip_tf_stderr.log")
    
    if result.returncode == 0:
        print("Tensorflow installed successfully!")
        
        # Test import
        try:
            import tensorflow as tf
            print(f"Tensorflow successfully imported. Version: {tf.__version__}")
        except ImportError as e:
            print(f"Installation succeeded but import failed: {e}")
            
except Exception as e:
    print(f"Script error: {e}")
