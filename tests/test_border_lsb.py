import pytest
from PIL import Image
from veil.border_lsb import (
    embed_message,
    extract_message,
)
from core.utils import calculate_border_capacity
from core.exceptions import CapacityError, ExtractionError


def make_image(size, color=(100, 150, 200)):
    return Image.new("RGB", size, color)


def test_border_lsb_roundtrip_simple():
    img = make_image((16, 16))
    message = b"hello border stego"

    stego = embed_message(img, message)
    extracted = extract_message(stego)

    assert extracted == message


@pytest.mark.parametrize("bpc", [1, 2, 3])
def test_border_lsb_roundtrip_various_bits(bpc):
    img = make_image((16, 16))
    message = b"border test"

    stego = embed_message(img, message, bits_per_channel=bpc)
    extracted = extract_message(stego, bits_per_channel=bpc)

    assert extracted == message


def test_border_lsb_capacity_too_small():
    img = make_image(size=(4, 4))
    # пикселей на границе = 4*4 - (4-2)*(4-2) = 16 - 4 = 12
    # capacity_bits = 12 * 3 * 1 = 36 -> 4 bytes
    message = b"12345"

    with pytest.raises(CapacityError):
        embed_message(img, message, bits_per_channel=1)


def test_border_lsb_not_enough_data_raises_extraction_error():
    img = make_image((0, 0))
    with pytest.raises(ExtractionError):
        extract_message(img)


def test_border_lsb_capacity_formula():
    img = make_image(size=(10, 8))

    cap = calculate_border_capacity(img, bits_per_channel=1, channels="R")

    # width=10, height=8
    # border pixels = 2*width + 2*(height-2) = 2*10 + 2*6 = 20+12=32
    # bits = 32 * 1 * 1 = 32 -> 4 bytes
    assert cap == 4
