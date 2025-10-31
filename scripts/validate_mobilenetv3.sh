#!/bin/bash
# Validate MobileNet V3 weights on ImageNet-1K validation set

echo "Validating MobileNet V3 Large..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_mobilenet_v3_large.yaml

echo ""
echo "Validating MobileNet V3 Small..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_mobilenet_v3_small.yaml
