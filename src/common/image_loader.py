from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms


def load_image_tensor(
    image_path,
    image_size=None,
    color_mode="RGB"
):
    """
    Load an image and convert it into a PyTorch tensor.

    Supported modes:
        RGB: 3 channels
        L:   1 grayscale channel

    image_size=None preserves original dimensions and pixel values.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if color_mode not in ("RGB", "L"):
        raise ValueError("color_mode must be 'RGB' or 'L'")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert(color_mode)

        if image_size is not None:
            image = image.resize(
                (image_size[1], image_size[0]),
                Image.Resampling.NEAREST
            )

        tensor = transforms.ToTensor()(image)

    return tensor


def get_image_info(tensor):
    """Return basic information about an image tensor."""

    return {
        "shape": tuple(tensor.shape),
        "channels": tensor.shape[0],
        "height": tensor.shape[1],
        "width": tensor.shape[2],
        "min": tensor.min().item(),
        "max": tensor.max().item()
    }
