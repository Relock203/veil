import pytest
from PIL import Image
from veil.lsb_matching import embed_message, extract_message
from core.utils import calculate_capacity
from core.exceptions import CapacityError


def test_lsb_matching_roundtrip_simple():
    image = Image.new("RGB", (50, 50), color="white")
    message = b"hello lsb matching"

    stego = embed_message(
        image=image,
        message=message,
        bits_per_channel=1,
        channels="RGB",
    )

    extracted = extract_message(
        image=stego,
        bits_per_channel=1,
        channels="RGB",
    )

    assert extracted == message


@pytest.mark.parametrize("channels", ["R", "G", "B", "RG", "RB", "GB", "RGB"])
def test_lsb_matching_roundtrip_different_channels(channels):
    image = Image.new("RGB", (60, 60), color="white")
    message = "Привет, Veil Matching!".encode("utf-8")

    stego = embed_message(
        image=image,
        message=message,
        bits_per_channel=1,
        channels=channels,
    )

    extracted = extract_message(
        image=stego,
        bits_per_channel=1,
        channels=channels,
    )

    assert extracted == message


def test_lsb_matching_raises_capacity_error_on_too_large_message():
    image = Image.new("RGB", (10, 10), color="white")
    capacity = calculate_capacity(image, bits_per_channel=1, channels="RGB")

    too_large_message = b"a" * (capacity + 1)

    with pytest.raises(CapacityError):
        embed_message(
            image=image,
            message=too_large_message,
            bits_per_channel=1,
            channels="RGB",
        )


def test_lsb_matching__embed_raises_on_unsupported_channels():
    image = Image.new("RGB", (60, 60), color="white")
    message = "Привет, Veil!".encode("utf-8")

    with pytest.raises(ValueError):
        embed_message(
            image,
            message=message,
            bits_per_channel=1,
            channels="CMYK",
        )