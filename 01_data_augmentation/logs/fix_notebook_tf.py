import json
import os

path = '/home/tej/code/tobacco/Copy of Integration of Traditional Knowledge and Modern Science/Integration of Traditional Knowledge and Modern Science/Integration of Traditional Knowledge M1/Agumentation.ipynb'

with open(path, 'r') as f:
    data = json.load(f)

# Create the installation cell
install_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# Run this cell to install TensorFlow safely.\n",
  "# The '--no-cache-dir' flag prevents out-of-memory errors which commonly cause installation failures.\n",
  "# 'tensorflow-cpu' is heavily recommended since we do 'CPU inference locally' and it is much faster to install without CUDA packages.\n",
  "# Feel free to change it to 'tensorflow' if you actually have a GPU setup.\n",
  "import sys\n",
  "!{sys.executable} -m pip install --no-cache-dir tensorflow"
 ]
}

# Prepend if not already there
if not any("pip install" in str(c.get("source")) for c in data['cells']):
    data['cells'].insert(0, install_cell)

with open(path, 'w') as f:
    json.dump(data, f, indent=1)

# Write to bash logs as requested
os.makedirs("logs/bash_logs", exist_ok=True)
with open("logs/bash_logs/tf_patch.md", "a") as f:
    f.write("Successfully patched Agumentation.ipynb with an installation cell for TensorFlow.\n")
