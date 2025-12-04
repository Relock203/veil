from __future__ import annotations

from typing import List

from PIL import Image

from core.utils import bytes_to_bits, bits_to_bytes,calculate_alpha_capacity
from core.exceptions import CapacityError, ExtractionError


def embed_message(
    image: Image.Image,
    message: bytes,
    bits_per_channel: int = 1,
) -> Image.Image:
    """Embed message into alpha channel using LSB (or multi-bit) embedding.

    Встраивается: 4 байта длины сообщения (big-endian) + само сообщение.

    Args:
        image: Source image (any mode, will be converted to RGBA).
        message: Bytes to hide.
        bits_per_channel: Number of least significant bits in alpha channel
            used for embedding (1..8).

    Returns:
        New Image with hidden data (mode RGBA).

    Raises:
        ValueError: If bits_per_channel is out of [1, 8].
        CapacityError: If message doesn't fit into alpha channel.
    """
    if not (1 <= bits_per_channel <= 8):
        raise ValueError("bits_per_channel must be between 1 and 8")

    capacity = calculate_alpha_capacity(image, bits_per_channel)

    length = len(message)
    length_bytes = length.to_bytes(4, byteorder="big")
    payload = length_bytes + message

    if len(payload) > capacity:
        raise CapacityError("Message is too large for alpha-channel capacity")

    payload_bits = bytes_to_bits(payload)
    total_bits = len(payload_bits)

    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size

    bit_index = 0

    for y in range(height):
        for x in range(width):
            if bit_index >= total_bits:
                return rgba

            r, g, b, a = pixels[x, y]

            remaining = total_bits - bit_index
            if remaining >= bits_per_channel:
                chunk = payload_bits[bit_index:bit_index + bits_per_channel]
            else:
                chunk = payload_bits[bit_index:total_bits] + [0] * (
                    bits_per_channel - remaining
                )

            # превращаем список битов в число 0..(2^bits_per_channel - 1)
            value = int("".join(str(bv) for bv in chunk), 2)

            mask = (1 << bits_per_channel) - 1
            new_a = (a & ~mask) | value

            pixels[x, y] = (r, g, b, new_a)
            bit_index += bits_per_channel

    return rgba


def extract_message(
    image: Image.Image,
    bits_per_channel: int = 1,
) -> bytes:
    """Extract message hidden in alpha channel using LSB-like scheme.

    Ожидается формат: 4 байта длины (big-endian) + сообщение.

    Args:
        image: Image with hidden data (any mode, will be converted to RGBA).
        bits_per_channel: Number of bits per alpha-channel used in embedding.

    Returns:
        Extracted payload (bytes).

    Raises:
        ValueError: If bits_per_channel is out of [1, 8].
        ExtractionError: If there is not enough data to read length or message.
    """
    if not (1 <= bits_per_channel <= 8):
        raise ValueError("bits_per_channel must be between 1 and 8")

    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size

    bits: List[int] = []
    mask = (1 << bits_per_channel) - 1

    for y in range(height):
        for x in range(width):
            _, _, _, a = pixels[x, y]
            value = a & mask
            bitstring = format(value, f"0{bits_per_channel}b")
            bits.extend(int(b) for b in bitstring)

    if len(bits) < 32:
        raise ExtractionError("Not enough data to read message length")

    length_bits = bits[:32]
    length_bytes = bits_to_bytes(length_bits)
    length = int.from_bytes(length_bytes, byteorder="big")

    needed_bits = length * 8
    total_needed = 32 + needed_bits

    if len(bits) < total_needed:
        raise ExtractionError("Not enough data to read full message")

    message_bits = bits[32:total_needed]
    message_bytes = bits_to_bytes(message_bits)
    return message_bytes
