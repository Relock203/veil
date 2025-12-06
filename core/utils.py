from PIL import Image
from typing import List, Tuple
import math


def bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        bit_string = bin(byte)[2:].zfill(8)
        bits.extend(int(b) for b in bit_string)

    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be a multiple of 8.")

    result = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        byte_str = ''.join(str(b) for b in byte_bits)
        byte_int = int(byte_str, 2)
        result.append(byte_int)

    return bytes(result)


def calculate_capacity(image: Image.Image, bits_per_channel: int, channels: str) -> int:
    if bits_per_channel <= 0:
        raise ValueError("bits_per_channel must be positive")

    width, height = image.size
    num_pixels = width * height
    num_channels = len(channels)
    total_bits = num_pixels * bits_per_channel * num_channels
    return total_bits // 8

PVD_RANGES = [
    (0, 7),
    (8, 15),
    (16, 31),
    (32, 63),
    (64, 127),
    (128, 255),
]


def get_range_info(d: int) -> tuple[int, int, int]:
    for lower, upper in PVD_RANGES:
        if lower <= d <= upper:
            width = upper - lower + 1
            # k = floor(log2(width))
            k = int(math.floor(math.log2(width)))
            return lower, upper, k

    lower, upper = PVD_RANGES[-1]
    width = upper - lower + 1
    k = int(math.floor(math.log2(width)))
    return lower, upper, k


def calculate_pvd_capacity(image, channel="R"):
    rgb = image.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size

    capacity_bits = 0

    for y in range(height):
        for x in range(0, width - 1, 2):
            r0, g0, b0 = pixels[x, y]
            r1, g1, b1 = pixels[x+1, y]

            if channel == "R":
                v0, v1 = r0, r1
            elif channel == "G":
                v0, v1 = g0, g1
            else:
                v0, v1 = b0, b1

            d = abs(v1 - v0)
            lower, upper, k = get_range_info(d)
            capacity_bits += k

    return capacity_bits // 8



def choose_target_difference(
        d: int,
        lower: int,
        upper: int,
        k: int,
        b_val: int
) -> int:
    mod = 2 ** k
    res = []
    for cand in range(lower, upper + 1):
        if cand % mod == b_val:
            d1 = abs(d - cand)
            res.append((d1, cand))
    return min(res)[1]


def calculate_alpha_capacity(image: Image.Image, bits_per_channel: int = 1) -> int:
    if not (1 <= bits_per_channel <= 8):
        raise ValueError("bits_per_channel must be between 1 and 8")

    rgba = image.convert("RGBA")
    width, height = rgba.size
    num_pixels = width * height

    total_bits = num_pixels * bits_per_channel
    capacity_bytes = total_bits // 8
    return capacity_bytes


def border_coordinates(width: int, height: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []

    if width <= 0 or height <= 0:
        return coords

    if width == 1 and height == 1:
        return [(0, 0)]

    if width == 1:
        for y in range(height):
            coords.append((0, y))
        return coords

    if height == 1:
        for x in range(width):
            coords.append((x, 0))
        return coords

    for x in range(width):
        coords.append((x, 0))

    for y in range(1, height - 1):
        coords.append((width - 1, y))

    for x in range(width - 1, -1, -1):
        coords.append((x, height - 1))

    for y in range(height - 2, 0, -1):
        coords.append((0, y))

    return coords


def calculate_border_capacity(
    image: Image.Image,
    bits_per_channel: int = 1,
    channels: str = "RGB",
) -> int:
    if bits_per_channel < 1:
        raise ValueError("bits_per_channel must be >= 1")

    rgb = image.convert("RGB")
    width, height = rgb.size

    border_coords = border_coordinates(width, height)
    num_pixels = len(border_coords)
    num_channels = len(channels)

    total_bits = num_pixels * num_channels * bits_per_channel
    capacity_bytes = total_bits // 8
    return capacity_bytes
