import pytest
from pathlib import Path
from PIL import Image
from veil.pvd import embed_message, extract_message
from core.utils import calculate_pvd_capacity
from core.exceptions import CapacityError


TEST_DIR = Path(__file__).parent
IMAGE_PATH = TEST_DIR / "data" / "pvd_gradient.png"


def load_gradient_image() -> Image.Image:
    img = Image.open(IMAGE_PATH).convert("RGB")
    return img


def test_pvd_roundtrip_gradient_simple():
    image = load_gradient_image()
    message = b"hello pvd stego"

    stego = embed_message(
        image=image,
        message=message,
        channels="R"
    )

    extracted = extract_message(
        image=stego,
        channels="R"
    )

    assert extracted == message


@pytest.mark.parametrize("channels", ["R", "G", "B", "RGB"])
def test_pvd_roundtrip_different_channels(channels: str):
    image = load_gradient_image()
    message = b"pvd multi-channel test"

    stego = embed_message(
        image=image,
        message=message,
        channels=channels
    )

    extracted = extract_message(
        image=stego,
        channels=channels
    )

    assert extracted == message


def test_pvd_raises_capacity_error_on_too_large_message():
    image = load_gradient_image()

    capacity = calculate_pvd_capacity(image, channel="R")

    too_large_message = b"A" * (capacity + 1)

    with pytest.raises(CapacityError):
        embed_message(
            image=image,
            message=too_large_message,
            channels="R"
        )


def test_pvd_embed_raises_on_unsupported_channels():
    image = load_gradient_image()
    message = "Привет, Veil!".encode("utf-8")

    with pytest.raises(ValueError):
        embed_message(
            image,
            message=message,
            channels="CMYK"
        )


def test_pvd_roundtrip_empty_message():
    image = load_gradient_image()
    message = b""

    stego = embed_message(image=image, message=message, channels="R")
    extracted = extract_message(image=stego, channels="R")

    assert extracted == message