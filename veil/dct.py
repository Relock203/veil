from __future__ import annotations
from typing import List, Tuple
from math import cos, pi
from PIL import Image
from core.utils import bytes_to_bits, bits_to_bytes
from core.exceptions import CapacityError, ExtractionError


BLOCK_SIZE = 8

CANDIDATE_POSITIONS: List[Tuple[int, int]] = [
    (2, 3),
    (3, 2),
    (4, 1),
]


def _image_to_luma_blocks(image: Image.Image) -> Tuple[List[List[List[float]]], int, int]:
    luma = image.convert("L")
    width, height = luma.size

    padded_width = (width + 7) // 8 * 8
    padded_height = (height + 7) // 8 * 8

    padded = Image.new("L", (padded_width, padded_height))
    padded.paste(luma, (0, 0))

    pixels = padded.load()
    blocks: List[List[List[float]]] = []

    for y in range(0, padded_height, 8):
        for x in range(0, padded_width, 8):
            block = [
                [float(pixels[x + dx, y + dy]) for dx in range(8)]
                for dy in range(8)
            ]
            blocks.append(block)

    return blocks, width, height


def _luma_blocks_to_image(blocks: List[List[List[float]]], width: int, height: int) -> Image.Image:
    padded_width = (width + 7) // 8 * 8
    padded_height = (height + 7) // 8 * 8

    img = Image.new("L", (padded_width, padded_height))
    pixels = img.load()

    idx = 0
    for y in range(0, padded_height, 8):
        for x in range(0, padded_width, 8):
            block = blocks[idx]
            idx += 1

            for dy in range(8):
                for dx in range(8):
                    val = round(block[dy][dx])
                    val = max(0, min(255, int(val)))
                    pixels[x + dx, y + dy] = val

    img = img.crop((0, 0, width, height))
    return img


def _dct_2d(block: List[List[float]]) -> List[List[float]]:
    n = BLOCK_SIZE  # 8
    result = [[0.0 for _ in range(n)] for _ in range(n)]

    def alpha(k: int) -> float:
        return 1 / (2 ** 0.5) if k == 0 else 1.0

    for v in range(n):
        for u in range(n):
            sum_val = 0.0
            for y in range(n):
                for x in range(n):
                    sum_val += (
                        block[y][x]
                        * cos((2 * x + 1) * u * pi / (2 * n))
                        * cos((2 * y + 1) * v * pi / (2 * n))
                    )

            result[v][u] = 0.25 * alpha(u) * alpha(v) * sum_val

    return result


def _idct_2d(coeffs: List[List[float]]) -> List[List[float]]:
    n = BLOCK_SIZE
    result = [[0.0 for _ in range(n)] for _ in range(n)]

    def alpha(k: int) -> float:
        return 1 / (2 ** 0.5) if k == 0 else 1.0

    for y in range(n):
        for x in range(n):
            sum_val = 0.0
            for v in range(n):
                for u in range(n):
                    sum_val += (
                            alpha(u)
                            * alpha(v)
                            * coeffs[v][u]
                            * cos((2 * x + 1) * u * pi / (2 * n))
                            * cos((2 * y + 1) * v * pi / (2 * n))
                    )

            result[y][x] = 0.25 * sum_val
    return result


def calculate_dct_capacity(image: Image.Image) -> int:
    blocks = _image_to_luma_blocks(image)[0]
    num_blocks = len(blocks)
    total_bits = num_blocks * len(CANDIDATE_POSITIONS)
    capacity_bytes = total_bits // 8
    return capacity_bytes


def embed_message(
        image: Image.Image,
        message: bytes,
) -> Image.Image:
    ycbcr = image.convert("YCbCr")
    Y, Cb, Cr = ycbcr.split()

    capacity = calculate_dct_capacity(image)

    length = len(message)
    header = length.to_bytes(4, byteorder="big")
    header3 = header * 3  # 12 байт = 96 бит

    payload = header3 + message

    if len(payload) > capacity:
        raise CapacityError("Message is too large for DCT capacity of this image")

    payload_bits = bytes_to_bits(payload)

    blocks, width, height = _image_to_luma_blocks(Y)

    bit_index = 0
    total_bits = len(payload_bits)

    new_blocks: List[List[List[float]]] = []

    for block in blocks:
        coeffs = _dct_2d(block)

        if bit_index < total_bits:
            for (v, u) in CANDIDATE_POSITIONS:
                if bit_index >= total_bits:
                    break

                bit = payload_bits[bit_index]

                val = coeffs[v][u]
                ival = int(round(val))
                ival = (ival & ~1) | bit
                coeffs[v][u] = float(ival)

                bit_index += 1

        new_blocks.append(coeffs)

    spatial_blocks: List[List[List[float]]] = []
    for coeffs in new_blocks:
        spatial_block = _idct_2d(coeffs)
        spatial_blocks.append(spatial_block)

    stego_Y = _luma_blocks_to_image(spatial_blocks, width, height)

    stego_ycbcr = Image.merge("YCbCr", (stego_Y, Cb, Cr))
    stego_rgb = stego_ycbcr.convert("RGB")
    return stego_rgb


def extract_message(
        image: Image.Image,
) -> bytes:
    ycbcr = image.convert("YCbCr")
    Y, _, _ = ycbcr.split()

    blocks, width, height = _image_to_luma_blocks(Y)

    bits: list[int] = []

    for block in blocks:
        coeffs = _dct_2d(block)

        for (v, u) in CANDIDATE_POSITIONS:
            val = coeffs[v][u]
            ival = int(round(val))
            lsb = ival & 1
            bits.append(lsb)

    if len(bits) < 96:
        raise ExtractionError("Not enough data to read header")

    header_bits = bits[:96]
    h1_bits = header_bits[0:32]
    h2_bits = header_bits[32:64]
    h3_bits = header_bits[64:96]

    h1 = int.from_bytes(bits_to_bytes(h1_bits), "big")
    h2 = int.from_bytes(bits_to_bytes(h2_bits), "big")
    h3 = int.from_bytes(bits_to_bytes(h3_bits), "big")

    print("DEBUG h1, h2, h3:", h1, h2, h3)
    print("DEBUG first 64 bits:", bits[:64])
    candidates = [h1, h2, h3]
    # ищем значение, которое встречается хотя бы два раза
    length = None
    for val in candidates:
        if candidates.count(val) >= 2:
            length = val
            break

    if length is None:
        raise ExtractionError("Cannot reliably decode message length")

    needed_bits = length * 8
    total_needed = 96 + needed_bits

    if len(bits) < total_needed:
        raise ExtractionError("Not enough data to read full message")

    message_bits = bits[96:total_needed]
    message_bytes = bits_to_bytes(message_bits)
    return message_bytes
