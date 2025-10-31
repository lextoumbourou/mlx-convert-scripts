#!/bin/bash
# Validate MobileNet V2 weights on ImageNet-1K validation set

echo "Validating MobileNet V2..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_mobilenet_v2.yaml
