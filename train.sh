echo "Installing requirements..."
pip install -r requirements.txt
echo "Getting data..."
sh data/raw/get_filtered.sh
echo "Preprocessing data..."
python src/data/data_creation.py
rm data/raw/filtered_paranmt.zip
echo "Training model..."
python src/models/train_model.py
echo "Done!"
