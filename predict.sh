MODEL_DIR="models/"
MODEL_ZIP="models/t5-best-model.zip"
MODEL_URL="http://bibatalov.ru:1234/model.zip"

# Check if the model directory exists and is not empty
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "Model directory already exists and is not empty, skipping download."
else
    mkdir models
    echo "Downloading model..."
    curl -o "$MODEL_ZIP" "$MODEL_URL"
    echo "Unzipping..."
    unzip -o "$MODEL_ZIP" -d "$MODEL_DIR"
fi

echo "Running..."
python src/models/predict_model.py "$1"