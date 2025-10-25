"""Convert EfficientNet PyTorch weights to MLX format.

This script downloads PyTorch EfficientNet weights (or uses existing ones),
loads them into an MLX EfficientNet model, then saves the MLX model's weights
as safetensors.
"""
import argparse
import sys
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-image/src"))

from mlxim.model.efficientnet._factory import efficientnet_b0


def load_pytorch_weights(model, weights_path: Union[str, Path]) -> None:
    """Load weights from PyTorch safetensors format.

    Args:
        weights_path: Path to PyTorch safetensors weights file
    """
    from safetensors import safe_open

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Load PyTorch weights
    pt_weights = {}
    with safe_open(weights_path, framework="numpy") as f:
        for key in f.keys():
            pt_weights[key] = f.get_tensor(key)

    # Helper function to transpose conv weights from PyTorch (O, I, H, W) to MLX (O, H, W, I)
    def transpose_conv(w):
        """Transpose from PyTorch (out, in, h, w) to MLX (out, h, w, in)."""
        return mx.array(w).transpose(0, 2, 3, 1)

    # Helper function to load weights for a Conv2dNormActivation layer
    def load_conv_norm_block(mlx_layer, pt_prefix):
        """Load weights for a Conv2dNormActivation block."""
        # Conv weights
        if f"{pt_prefix}.0.weight" in pt_weights:
            mlx_layer.layers[0].weight = transpose_conv(pt_weights[f"{pt_prefix}.0.weight"])
            if f"{pt_prefix}.0.bias" in pt_weights:
                mlx_layer.layers[0].bias = mx.array(pt_weights[f"{pt_prefix}.0.bias"])

        # BatchNorm weights
        if f"{pt_prefix}.1.weight" in pt_weights:
            mlx_layer.layers[1].weight = mx.array(pt_weights[f"{pt_prefix}.1.weight"])
            mlx_layer.layers[1].bias = mx.array(pt_weights[f"{pt_prefix}.1.bias"])
            mlx_layer.layers[1].running_mean = mx.array(pt_weights[f"{pt_prefix}.1.running_mean"])
            mlx_layer.layers[1].running_var = mx.array(pt_weights[f"{pt_prefix}.1.running_var"])

    # Load stem (features.0)
    load_conv_norm_block(model.features[0], "features.0")

    # Load blocks (features.1 through features.7 for B0)
    stage_idx = 1
    for stage_blocks in model.features[1:-1]:  # Exclude stem and head
        for layer_idx, block in enumerate(stage_blocks):
            pt_block_prefix = f"features.{stage_idx}.{layer_idx}.block"

            # Determine if expansion layer exists by checking the weights
            has_expansion = f"{pt_block_prefix}.1.0.weight" in pt_weights

            mlx_layer_idx = 0

            if has_expansion:
                # Has expansion: 0=expand, 1=depthwise, 2=SE, 3=project
                # Expansion layer
                load_conv_norm_block(block.block[mlx_layer_idx], f"{pt_block_prefix}.0")
                mlx_layer_idx += 1

                # Depthwise layer
                load_conv_norm_block(block.block[mlx_layer_idx], f"{pt_block_prefix}.1")
                mlx_layer_idx += 1

                # SE layer
                se_layer = block.block[mlx_layer_idx]
                pt_se_prefix = f"{pt_block_prefix}.2"
                if f"{pt_se_prefix}.fc1.weight" in pt_weights:
                    se_layer.fc1.weight = transpose_conv(pt_weights[f"{pt_se_prefix}.fc1.weight"])
                    se_layer.fc1.bias = mx.array(pt_weights[f"{pt_se_prefix}.fc1.bias"])
                    se_layer.fc2.weight = transpose_conv(pt_weights[f"{pt_se_prefix}.fc2.weight"])
                    se_layer.fc2.bias = mx.array(pt_weights[f"{pt_se_prefix}.fc2.bias"])
                mlx_layer_idx += 1

                # Projection layer
                load_conv_norm_block(block.block[mlx_layer_idx], f"{pt_block_prefix}.3")
            else:
                # No expansion: 0=depthwise, 1=SE, 2=project
                # Depthwise layer
                load_conv_norm_block(block.block[mlx_layer_idx], f"{pt_block_prefix}.0")
                mlx_layer_idx += 1

                # SE layer
                se_layer = block.block[mlx_layer_idx]
                pt_se_prefix = f"{pt_block_prefix}.1"
                if f"{pt_se_prefix}.fc1.weight" in pt_weights:
                    se_layer.fc1.weight = transpose_conv(pt_weights[f"{pt_se_prefix}.fc1.weight"])
                    se_layer.fc1.bias = mx.array(pt_weights[f"{pt_se_prefix}.fc1.bias"])
                    se_layer.fc2.weight = transpose_conv(pt_weights[f"{pt_se_prefix}.fc2.weight"])
                    se_layer.fc2.bias = mx.array(pt_weights[f"{pt_se_prefix}.fc2.bias"])
                mlx_layer_idx += 1

                # Projection layer
                load_conv_norm_block(block.block[mlx_layer_idx], f"{pt_block_prefix}.2")

        stage_idx += 1

    # Load head (last feature layer)
    load_conv_norm_block(model.features[-1], f"features.{stage_idx}")

    # Load classifier
    if "classifier.1.weight" in pt_weights:
        # PyTorch Linear: (out_features, in_features)
        # MLX Linear: (out_features, in_features) - same!
        model.classifier.layers[1].weight = mx.array(pt_weights["classifier.1.weight"])
        model.classifier.layers[1].bias = mx.array(pt_weights["classifier.1.bias"])

    print(f"Successfully loaded PyTorch weights from {weights_path}")


def convert_pytorch_to_mlx(pytorch_weights_path: Path, output_path: Path, model_name: str = "efficientnet_b0"):
    """Convert PyTorch EfficientNet weights to MLX format.

    Args:
        pytorch_weights_path: Path to PyTorch safetensors file
        output_path: Path to save MLX-formatted weights
        model_name: Name of the model architecture
    """
    print(f"Creating MLX {model_name} model...")

    # Create MLX model
    if model_name == "efficientnet_b0":
        model = efficientnet_b0()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Loading PyTorch weights from {pytorch_weights_path}...")
    load_pytorch_weights(model, pytorch_weights_path)

    print(f"Saving MLX weights to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten and save MLX model weights (same method as in _utils.py)
    from mlx.utils import tree_flatten

    weights_dict = dict(tree_flatten(model.parameters()))

    mx.save_safetensors(str(output_path), weights_dict)

    print(f"✓ Successfully saved MLX weights to {output_path}")
    print(f"  Saved {len(weights_dict)} weight tensors")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def download_pytorch_weights(model_name: str, weights_dir: Path) -> Path:
    """Download PyTorch weights from torchvision if not already present.

    Args:
        model_name: Name of the model (e.g., 'efficientnet_b0')
        weights_dir: Directory to save weights

    Returns:
        Path to the downloaded weights file
    """
    output_path = weights_dir / f"{model_name}_torch.safetensors"

    if output_path.exists():
        print(f"PyTorch weights already exist at {output_path}")
        return output_path

    print(f"Downloading {model_name} weights from torchvision...")

    try:
        import torch
        import torchvision.models as models
        from safetensors.torch import save_file as torch_save_file

        # Get the model and download weights
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        elif model_name == "efficientnet_b1":
            model = models.efficientnet_b1(weights="IMAGENET1K_V2")
        elif model_name == "efficientnet_b2":
            model = models.efficientnet_b2(weights="IMAGENET1K_V1")
        elif model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        elif model_name == "efficientnet_b4":
            model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        elif model_name == "efficientnet_b5":
            model = models.efficientnet_b5(weights="IMAGENET1K_V1")
        elif model_name == "efficientnet_b6":
            model = models.efficientnet_b6(weights="IMAGENET1K_V1")
        elif model_name == "efficientnet_b7":
            model = models.efficientnet_b7(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Save as safetensors
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch_save_file(model.state_dict(), str(output_path))
        print(f"✓ Downloaded and saved PyTorch weights to {output_path}")

    except ImportError:
        raise ImportError("torch and torchvision are required to download weights. "
                        "Please install them or provide existing PyTorch weights.")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert EfficientNet PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=[
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
            "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"
        ],
        help="Model name to convert"
    )
    parser.add_argument(
        "--pytorch-weights",
        type=str,
        default=None,
        help="Path to existing PyTorch safetensors file (if not provided, will download)"
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "working"),
        help="Working directory for PyTorch weights"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "output"),
        help="Output directory for converted MLX weights"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download PyTorch weights from torchvision"
    )

    args = parser.parse_args()

    weights_dir = Path(args.working_dir)
    output_dir = Path(args.output_dir)

    # Get PyTorch weights path
    if args.pytorch_weights:
        pytorch_weights_path = Path(args.pytorch_weights)
        if not pytorch_weights_path.exists():
            raise FileNotFoundError(f"PyTorch weights not found: {pytorch_weights_path}")
    elif args.download:
        pytorch_weights_path = download_pytorch_weights(args.model, weights_dir)
    else:
        # Check if already exists
        pytorch_weights_path = weights_dir / f"{args.model}_torch.safetensors"
        if not pytorch_weights_path.exists():
            print(f"PyTorch weights not found at {pytorch_weights_path}")
            print("Use --download to download from torchvision or --pytorch-weights to specify path")
            return

    # Convert to MLX format
    mlx_weights_path = output_dir / f"{args.model}-mlxim" / "model.safetensors"
    convert_pytorch_to_mlx(pytorch_weights_path, mlx_weights_path, args.model)

    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"  PyTorch weights: {pytorch_weights_path}")
    print(f"  MLX weights:     {mlx_weights_path}")
    print("="*60)


if __name__ == "__main__":
    main()
