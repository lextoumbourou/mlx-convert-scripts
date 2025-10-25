# MLX-image scripts

This is a collection of scripts that I've used to create mlx-vision datasets.

## Setup

Clone the `mlx-image` project to `./mlx-image`, and checkout the branch with the model changes.

```bash
git clone git@github.com:lextoumbourou/mlx-image.git
git checkout lex-efficientnet-vision
```

Clone `vision` for model comparisions:

```bash
git clone git@github.com:pytorch/vision.git
```

Setup uv

```
uv sync
```

Download the validation datasets:

```bash
# Download the ImageNet-1k dataset to datasets directory
uv run huggingface-cli download mlx-vision/imagenet-1k --repo-type dataset --local-dir datasets/imagenet-1k
```

## Convert EfficientNet weights

```bash
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b0 --download
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b1 --download
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b2 --download
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b3 --download
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b4 --download
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b5 --download
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b6 --download
uv run scripts/convert_efficientnet_weights.py --model efficientnet_b7 --download
```

## Test EfficientNet models

```bash
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b0.yaml
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b1.yaml
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b2.yaml
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b3.yaml
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b4.yaml
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b5.yaml
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b6.yaml
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b7.yaml
```

## Upload to HuggingFace

```bash
cd output/efficientnet_b0-mlxim && hf upload mlx-vision/efficientnet_b0-mlxim .
cd output/efficientnet_b1-mlxim && hf upload mlx-vision/efficientnet_b1-mlxim .
cd output/efficientnet_b2-mlxim && hf upload mlx-vision/efficientnet_b2-mlxim .
cd output/efficientnet_b3-mlxim && hf upload mlx-vision/efficientnet_b3-mlxim .
cd output/efficientnet_b4-mlxim && hf upload mlx-vision/efficientnet_b4-mlxim .
cd output/efficientnet_b5-mlxim && hf upload mlx-vision/efficientnet_b5-mlxim .
cd output/efficientnet_b6-mlxim && hf upload mlx-vision/efficientnet_b6-mlxim .
cd output/efficientnet_b7-mlxim && hf upload mlx-vision/efficientnet_b7-mlxim .
```
