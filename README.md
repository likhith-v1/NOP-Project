# NOP-Project

Training and comparison framework for the Lipschitz Momentum optimizer on medical image classification.

## Main Project: Retinal OCT

Retinal OCT is the active and maintained pipeline in this repository.

## Hardware Support

This project supports all three backends:

- cpu
- mps (Apple Silicon)
- cuda (NVIDIA GPU)

Choose your backend in the dataset config files:

- retinal OCT: `retinal_oct/configs/config.yaml`

Example:

```yaml
project:
  device: "cpu"   # or "mps" / "cuda"
```

## Quick Start

1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

2. Train one optimizer (default is lipschitz_momentum)

```bash
python3 retinal_oct/train.py
```

3. Run all configured optimizers + generate comparison plots

```bash
python3 retinal_oct/compare.py --config retinal_oct/configs/config.yaml
```

4. Regenerate plots only (without retraining)

```bash
python3 retinal_oct/compare.py --plot-only --config retinal_oct/configs/config.yaml
```

5. Evaluate a single checkpoint and export test plots

```bash
python3 retinal_oct/evaluate.py --optimizer lipschitz_momentum --config retinal_oct/configs/config.yaml
```

## Retinal OCT Script Flow

For the main project workflow:

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Optional: faster install with uv
# uv pip install -r requirements.txt

# Download the Retinal OCT dataset from your preferred source
# and place it under datasets/OCT in train/val/test class folders.

# Default active training run
python3 retinal_oct/train.py
```

## Retinal OCT Results Layout

- checkpoints: `results/retinal_oct/checkpoints`
- training logs: `results/retinal_oct/logs`
- evaluation/comparison plots: `results/retinal_oct/plots`

## Separate Section: Chest X-Ray (Deprecated)

Chest X-Ray is not the main pipeline and is kept only for reference/legacy runs.

Config file:

- `chest_xray_deprecated/configs/config.yaml`

Legacy commands:

```bash
python3 chest_xray_deprecated/train.py
python3 chest_xray_deprecated/compare.py --config chest_xray_deprecated/configs/config.yaml
```

Optional legacy dataset download:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d xray_data
```

Related deprecated paths:

- `chest_xray_deprecated/`
- `results/chest_xray_deprecated/`
- `datasets/xray_data/chest_xray_deprecated/`

## Data Sources

- Retinal OCT dataset: https://data.mendeley.com/datasets/rscbjbr9sj/3
- Chest X-Ray Pneumonia dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia