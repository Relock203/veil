import pytest
from PIL import Image

from veil.lsb import (
    embed_message,
    extract_message,
    calculate_capacity,
)
from core.exceptions import CapacityError


def test_lsb_embed_extract_roundtrip_simple():
    image = Image.new("RGB", (50, 50), color="white")
    message = "hello veil".encode("utf-8")

    stego = embed_message(
        image,
        message=message,
        bits_per_channel=1,
        channels="RGB",
    )

    extracted = extract_message(
        stego,
        bits_per_channel=1,
        channels="RGB",
    )

    assert extracted == message


@pytest.mark.parametrize(
    "bits_per_channel,channels",
    [
        (1, "RGB"),
        (2, "RGB"),
        (1, "R"),
        (2, "RG"),
    ],
)
def test_lsb_embed_extract_parametrized(bits_per_channel, channels):
    image = Image.new("RGB", (60, 60), color="white")
    message = "Привет, Veil!".encode("utf-8")

    stego = embed_message(
        image,
        message=message,
        bits_per_channel=bits_per_channel,
        channels=channels,
    )

    extracted = extract_message(
        stego,
        bits_per_channel=bits_per_channel,
        channels=channels,
    )

    assert extracted == message


def test_lsb_embed_raises_capacity_error_on_too_large_message():
    image = Image.new("RGB", (10, 10), color="white")  # маленькое изображение
    capacity = calculate_capacity(image, bits_per_channel=1, channels="RGB")

    too_large_message = b"a" * (capacity + 1)

    with pytest.raises(CapacityError):
        embed_message(
            image,
            message=too_large_message,
            bits_per_channel=1,
            channels="RGB",
        )


def test_lsb_embed_raises_on_unsupported_channels():
    image = Image.new("RGB", (60, 60), color="white")
    message = "Привет, Veil!".encode("utf-8")

    with pytest.raises(ValueError):
        embed_message(
            image,
            message=message,
            bits_per_channel=1,
            channels="CMYK",
        )