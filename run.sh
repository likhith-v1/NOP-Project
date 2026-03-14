#!/usr/bin/env bash
set -euo pipefail

# Script for training models
# Install dependencies
python -m pip install -r requirements.txt

# Or use uv (recommended for faster installs)
# uv pip install -r requirements.txt

# Kaggle Dataset Download (if not already downloaded)
# Kaggle python library and api token should be set up for this to work

# kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip the dataset (if not already unzipped)
# unzip chest-xray-pneumonia.zip -d xray_data

# kaggle datasets download -d paultimothymooney/kermany2018

# Unzip the dataset (if not already unzipped)
# unzip kermany2018.zip -d retinal-oct_data


# Start training (default: active dataset)
python retinal_oct/train.py

# Deprecated dataset (optional)
# python chest_xray_deprecated/train.py