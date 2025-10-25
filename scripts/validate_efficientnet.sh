#!/bin/bash

echo "Validating EfficientNet B0..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b0.yaml

echo ""
echo "Validating EfficientNet B1..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b1.yaml
