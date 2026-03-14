# NOP-Project

Training and comparison framework for the Lipschitz Momentum optimizer on medical image classification.

## Project Status

- Retinal OCT is the active pipeline.
- Chest X-Ray is kept for reference under a deprecated namespace.

## Hardware Support

This project supports all three backends:

- cpu
- mps (Apple Silicon)
- cuda (NVIDIA GPU)

Choose your backend in the dataset config files:

- retinal OCT: `retinal_oct/configs/config.yaml`
- deprecated chest x-ray: `chest_xray_deprecated/configs/config.yaml`

Example:

```yaml
project:
  device: "cpu"   # or "mps" / "cuda"
```

## Quick Start

1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

2. Train one optimizer (default is lipschitz_momentum)

```bash
python retinal_oct/train.py
```

3. Run all configured optimizers + generate comparison plots

```bash
python retinal_oct/compare.py --config retinal_oct/configs/config.yaml
```

4. Regenerate plots only (without retraining)

```bash
python retinal_oct/compare.py --plot-only --config retinal_oct/configs/config.yaml
```

5. Evaluate a single checkpoint and export test plots

```bash
python retinal_oct/evaluate.py --optimizer lipschitz_momentum --config retinal_oct/configs/config.yaml
```

## Full Script Flow (run.sh)

The shell script uses the sequence below:

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Optional: faster install with uv
# uv pip install -r requirements.txt

# Optional dataset download (requires Kaggle API setup)
# kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# unzip chest-xray-pneumonia.zip -d xray_data
# kaggle datasets download -d paultimothymooney/kermany2018
# unzip kermany2018.zip -d retinal-oct_data

# Default active training run
python retinal_oct/train.py

# Optional deprecated dataset run
# python chest_xray_deprecated/train.py
```

## Results Layout

- checkpoints: `results/retinal_oct/checkpoints`
- training logs: `results/retinal_oct/logs`
- evaluation/comparison plots: `results/retinal_oct/plots`

## Deprecated Pipeline

If needed, the old chest x-ray path is still runnable:

```bash
python chest_xray_deprecated/train.py
python chest_xray_deprecated/compare.py --config chest_xray_deprecated/configs/config.yaml
```

Related deprecated paths:

- `chest_xray_deprecated/`
- `results/chest_xray_deprecated/`
- `datasets/xray_data/chest_xray_deprecated/`

## Data Sources

- OCT2017 / Kermany2018: https://data.mendeley.com/datasets/rscbjbr9sj/3
- OCT exploration notebook/source: https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images
- Chest X-Ray Pneumonia dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia