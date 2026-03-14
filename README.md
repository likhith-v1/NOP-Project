# NOP-Project

## Dataset Status

- Retinal OCT: active
- Chest X-Ray: deprecated

## CUDA or MPS required

## Quick Start

1. Install dependencies:

	python -m pip install -r requirements.txt 

2. Train on the active dataset:

	python retinal_oct/train.py

3. Compare optimizers and regenerate plots from logs:

	python retinal_oct/compare.py --plot-only --config retinal_oct/configs/config.yaml

## Notes

- Chest X-Ray code and data are retained under the deprecated namespace:
  - chest_xray_deprecated/
  - results/chest_xray_deprecated/
  - datasets/xray_data/chest_xray_deprecated/

https://data.mendeley.com/datasets/rscbjbr9sj/3

https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia