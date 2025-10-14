
#do either:
#   bash install_ds.sh lg
#or
#   bash install_ds.sh


set -e

# git clone https://github.com/antoinsader/reramker_test.git




TARGET_DIR="data/raw"

mkdir -p "$TARGET_DIR"

echo 'USING LG DATASET'
unzip -o train_dictionary.zip -d  "$TARGET_DIR"
unzip -o traindev_.zip -d "$TARGET_DIR"
# unzip -o test.zip -d "$TARGET_DIR"





echo "Files successfully extracted to $TARGET_DIR"

python -m venv myenv
source myenv/bin/activate

# === Step 5: Install dependencies ===
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install faiss-gpu-cu12
pip install tqdm transformers requests psutil
pip install datasets

echo "Setup complete!"


