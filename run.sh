# Script for training the model
# Install Dependencies
pip install -r requirements.txt

# or use uv (recommeded for faster installs)
# uv pip install -r requirements.txt

# Kaggle Dataset Download (if not already downloaded)
# Kaggle python library and api token should be set up for this to work

# kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip the dataset (if not already unzipped)
# unzip chest-xray-pneumonia.zip -d xray_data

# Start training
python train.py