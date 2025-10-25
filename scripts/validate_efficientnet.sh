#!/bin/bash

echo "Validating EfficientNet B0..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b0.yaml

echo ""
echo "Validating EfficientNet B1..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b1.yaml

echo ""
echo "Validating EfficientNet B2..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b2.yaml

echo ""
echo "Validating EfficientNet B3..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b3.yaml

echo ""
echo "Validating EfficientNet B4..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b4.yaml

echo ""
echo "Validating EfficientNet B5..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b5.yaml

echo ""
echo "Validating EfficientNet B6..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b6.yaml

echo ""
echo "Validating EfficientNet B7..."
PYTHONPATH=mlx-image/src uv run python mlx-image/validation.py --config val_config/validation_efficientnet_b7.yaml
