"""Image loading and saving utilities for Veil."""

from pathlib import Path

from PIL import Image

from core.exceptions import UnsupportedFormatError


def load_image(path: Path) -> Image.Image:
    """Load an image from the given path.

    Args:
        path: Path to the image file.

    Returns:
        Loaded Pillow Image object in RGB mode.

    Raises:
        FileNotFoundError: If file does not exist.
        UnsupportedFormatError: If image format is unsupported.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        image = Image.open(path)
        image.load()
    except OSError as exc:
        raise UnsupportedFormatError(f"Cannot open image: {path}") from exc

    # Для единообразия всё приводим к RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def save_image(image: Image.Image, path: Path) -> None:
    """Save an image to the given path.

    Args:
        image: Pillow image object to save.
        path: Target path.

    Raises:
        OSError: If saving failed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)