import pytest
from PIL import Image

from veil.lsb import (
    embed_message,
    extract,
    calculate_capacity,
    embed_file_to_file,
    extract_from_file,
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

    extracted = extract(
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

    extracted = extract(
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


def test_lsb_file_to_file_roundtrip(tmp_path):
    image = Image.new("RGB", (64, 64), color="white")
    input_path = tmp_path / "cover.png"
    image.save(input_path)

    message = b"file-based test message"

    output_path = tmp_path / "stego.png"

    embed_file_to_file(
        input_path=input_path,
        output_path=output_path,
        message=message,
        bits_per_channel=2,
        channels="RGB",
    )

    extracted = extract_from_file(
        input_path=output_path,
        bits_per_channel=2,
        channels="RGB",
    )

    assert extracted == message