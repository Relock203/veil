import pytest
from PIL import Image

from veil.alpha import (
    embed_message,
    extract_message,
    calculate_alpha_capacity,
)
from core.exceptions import CapacityError, ExtractionError


def make_test_image(mode="RGBA", size=(32, 32), color=(10, 20, 30, 255)):
    """Create a simple uniform image for testing."""
    return Image.new(mode, size, color)


def test_alpha_roundtrip_default_bits():
    img = make_test_image()
    message = b"hello alpha stego!"

    stego = embed_message(img, message)
    extracted = extract_message(stego)

    assert extracted == message


@pytest.mark.parametrize("bpc", [1, 2, 3, 4])
def test_alpha_roundtrip_various_bits(bpc):
    img = make_test_image()
    message = b"alpha test payload"

    stego = embed_message(img, message, bits_per_channel=bpc)
    extracted = extract_message(stego, bits_per_channel=bpc)

    assert extracted == message


def test_alpha_capacity_too_small_raises():
    img = make_test_image(size=(4, 4))
    message = b"this is too long for 4x4"

    with pytest.raises(CapacityError):
        embed_message(img, message, bits_per_channel=1)


def test_alpha_extract_from_image_without_message():
    img = make_test_image()

    with pytest.raises(ExtractionError):
        extract_message(img)


def test_alpha_works_on_rgb_image():
    """Alpha stego must work even if the source image is RGB."""
    img = make_test_image(mode="RGB", size=(32, 32), color=(50, 100, 150))
    message = b"rgb alpha test"

    stego = embed_message(img, message, bits_per_channel=2)
    extracted = extract_message(stego, bits_per_channel=2)

    assert extracted == message


def test_alpha_capacity_correctness():
    img = make_test_image(size=(10, 10))
    cap = calculate_alpha_capacity(img, bits_per_channel=1)

    # 10 * 10 * 1 bit = 100 bits = 12.5 bytes → floor to 12 bytes
    assert cap == 12
