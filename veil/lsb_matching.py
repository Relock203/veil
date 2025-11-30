from PIL import Image
from core.exceptions import CapacityError, ExtractionError
from random import choice
from core.utils import bytes_to_bits, bits_to_bytes, calculate_capacity


def embed_message(
        image: Image.Image,
        message: bytes,
        bits_per_channel: int = 1,
        channels: str = 'RGB',
) -> Image.Image:
    if bits_per_channel != 1:
        raise ValueError("bits_per_channel must be 1 in this mode")

    capacity = calculate_capacity(image, 1, channels)

    length = len(message)
    length_bytes = length.to_bytes(4, byteorder='big')
    payload = length_bytes + message
    payload_bits = bytes_to_bits(payload)

    if len(payload) > capacity:
        raise CapacityError("Message is too large for given image and settings")

    stego = image.copy()
    pixels = stego.load()
    width, height = stego.size

    bit_index = 0
    total_bits = len(payload_bits)
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            values = [r, g, b]

            if bit_index >= total_bits:
                return stego

            for ch in channels:
                if bit_index >= total_bits:
                    break

                if ch == 'R':
                    idx = 0
                elif ch == 'G':
                    idx = 1
                elif ch == 'B':
                    idx = 2
                else:
                    continue

                value = values[idx]
                last_bit = value & 1
                current_bit = payload_bits[bit_index]
                if last_bit != current_bit:
                    if value == 0:
                        value = 1
                    elif value == 255:
                        value = 254
                    else:
                        value += choice([-1, +1])
                bit_index += 1
                values[idx] = value

            pixels[x, y] = tuple(values)
    return stego


def extract(
        image: Image.Image,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> bytes:
    if bits_per_channel != 1:
        raise ValueError("bits_per_channel must be 1 in this mode")

    pixels = image.load()
    width, height = image.size

    bits: list[int] = []

    for y in range(height):
        for x in range(width):
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

                value = values[idx]
                last_bit = value & 1
                bits.append(last_bit)

    if len(bits) < 32:
        raise ExtractionError("Not enough data to read message length")

    length_bits = bits[:32]
    length_bytes = bits_to_bytes(length_bits)
    length = int.from_bytes(length_bytes, byteorder="big")

    needed_bits = length * 8
    total_needed = 32 + needed_bits

    if len(bits) < total_needed:
        raise ExtractionError("Not enough embedded data for declared message length")

    message_bits = bits[32:total_needed]
    message_bytes = bits_to_bytes(message_bits)

    return message_bytes
