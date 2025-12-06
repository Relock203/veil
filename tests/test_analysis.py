import pytest
from PIL import Image

from veil.analysis.lsb_statistics import lsb_statistics
from veil.analysis.lsb_plane_image import lsb_plane_image


def make_solid_image(size=(16, 16), color=(0, 0, 0)):
    """Однородное изображение для тестов."""
    return Image.new("RGB", size, color)


def make_pattern_image(size=(8, 8)):
    """Создаём картинку, где LSB по R чередуются 0/1."""
    img = Image.new("RGB", size)
    pixels = img.load()
    width, height = size

    for y in range(height):
        for x in range(width):
            r = 0 if (x % 2 == 0) else 1
            g = 0
            b = 0
            pixels[x, y] = (r, g, b)
    return img


def test_lsb_statistics_all_zero_lsb():
    img = make_solid_image(color=(0, 0, 0))

    stats = lsb_statistics(img, channels="R")

    r_stats = stats["R"]
    assert r_stats["ones"] == 0
    assert r_stats["zeros"] == r_stats["total"]
    assert r_stats["p1"] == 0.0
    assert r_stats["p0"] == 1.0


def test_lsb_statistics_balanced_pattern():
    img = make_pattern_image(size=(8, 8))  # 8*8 = 64 пикселя
    stats = lsb_statistics(img, channels="R")

    r_stats = stats["R"]
    assert r_stats["total"] == 64
    assert r_stats["zeros"] == 32
    assert r_stats["ones"] == 32
    assert pytest.approx(r_stats["p0"], rel=1e-6) == 0.5
    assert pytest.approx(r_stats["p1"], rel=1e-6) == 0.5
    assert pytest.approx(r_stats["chi2"], abs=1e-6) == 0.0


def test_lsb_statistics_multiple_channels():
    img = make_solid_image(color=(1, 2, 3))  # R=1(LSB=1), G=2(0), B=3(1)
    stats = lsb_statistics(img, channels="RGB")

    r_stats = stats["R"]
    g_stats = stats["G"]
    b_stats = stats["B"]

    assert r_stats["ones"] == r_stats["total"]
    assert g_stats["zeros"] == g_stats["total"]
    assert b_stats["ones"] == b_stats["total"]


def test_lsb_plane_image_mode_and_size():
    img = make_pattern_image(size=(10, 6))
    plane = lsb_plane_image(img, channel="R")

    assert plane.mode == "L"
    assert plane.size == img.size


def test_lsb_plane_image_values():
    img = make_pattern_image(size=(4, 1))  # R: 0,1,0,1
    plane = lsb_plane_image(img, channel="R")
    pixels = plane.load()

    assert pixels[0, 0] == 0
    assert pixels[1, 0] == 255
    assert pixels[2, 0] == 0
    assert pixels[3, 0] == 255
