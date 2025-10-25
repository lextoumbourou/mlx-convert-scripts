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
