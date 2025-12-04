from typing import List

from PIL import Image

from core.utils import bytes_to_bits, bits_to_bytes, calculate_border_capacity, border_coordinates
from core.exceptions import CapacityError, ExtractionError


def embed_message(
    image: Image.Image,
    message: bytes,
    bits_per_channel: int = 1,
    channels: str = "RGB",
) -> Image.Image:
    """Embed message using LSB in border pixels only.

    Формат полезной нагрузки:
    - 4 байта длины (big-endian)
    - сами данные

    Args:
        image: Source image (any mode, converted to RGB internally).
        message: Bytes to hide.
        bits_per_channel: How many LSBs to use per channel (>=1).
        channels: Which color channels to use, subset of "RGB".

    Returns:
        New Image with hidden data (mode RGB).

    Raises:
        CapacityError: If message does not fit into border capacity.
    """
    if bits_per_channel < 1:
        raise ValueError("bits_per_channel must be >= 1")

    capacity = calculate_border_capacity(image, bits_per_channel, channels)

    length = len(message)
    length_bytes = length.to_bytes(4, byteorder="big")
    payload = length_bytes + message

    if len(payload) > capacity:
        raise CapacityError("Message is too large for border-LSB capacity")

    payload_bits = bytes_to_bits(payload)
    total_bits = len(payload_bits)
    bit_index = 0

    rgb = image.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size

    border_coords = border_coordinates(width, height)

    for (x, y) in border_coords:
        if bit_index >= total_bits:
            break

        r, g, b = pixels[x, y]
        values = [r, g, b]

        for ch in channels:
            if bit_index >= total_bits:
                break

            if ch == "R":
                idx = 0
            elif ch == "G":
                idx = 1
            elif ch == "B":
                idx = 2
            else:
                continue

            remaining = total_bits - bit_index
            if remaining >= bits_per_channel:
                chunk = payload_bits[bit_index:bit_index + bits_per_channel]
            else:
                chunk = payload_bits[bit_index:total_bits] + [0] * (
                    bits_per_channel - remaining
                )

            value_bits = int("".join(str(bv) for bv in chunk), 2)
            mask = (1 << bits_per_channel) - 1
            values[idx] = (values[idx] & ~mask) | value_bits

            bit_index += bits_per_channel

        pixels[x, y] = tuple(values)

    return rgb


def extract_message(
    image: Image.Image,
    bits_per_channel: int = 1,
    channels: str = "RGB",
) -> bytes:
    """Extract message hidden in border pixels using LSB.

    Args:
        image: Image with hidden data (any mode, converted to RGB internally).
        bits_per_channel: Same as used in embedding.
        channels: Same channels as used in embedding.

    Returns:
        Extracted payload bytes.

    Raises:
        ExtractionError: If cannot read length or full message.
    """
    if bits_per_channel < 1:
        raise ValueError("bits_per_channel must be >= 1")

    rgb = image.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size

    border_coords = border_coordinates(width, height)
    bits: List[int] = []

    for (x, y) in border_coords:
        r, g, b = pixels[x, y]
        values = [r, g, b]

        for ch in channels:
            if ch == "R":
                idx = 0
            elif ch == "G":
                idx = 1
            elif ch == "B":
                idx = 2
            else:
                continue

            mask = (1 << bits_per_channel) - 1
            value = values[idx] & mask
            bitstring = format(value, f"0{bits_per_channel}b")
            bits.extend(int(bv) for bv in bitstring)

    # Нужно минимум 32 бита для длины
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
