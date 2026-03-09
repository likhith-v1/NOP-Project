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

# kaggle datasets download -d paultimothymooney/kermany2018

# Unzip the dataset (if not already unzipped)
# unzip kermany2018.zip -d retinal-oct_data


# Start training (default: chest_xray)
python chest_xray/train.py