# echo "Downloading model..."
# curl -o models/t5-best-model.zip http://bibatalov.ru:1234/model.zip
# echo "Unzipping..."
# unzip models/t5-best-model.zip -d models/t5-best-model
# echo "Running..."
python src/models/predict_model.py "$1"