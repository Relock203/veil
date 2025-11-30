from PIL import Image
from random import choice
from core.exceptions import CapacityError, ExtractionError
from core.utils import bytes_to_bits, bits_to_bytes, calculate_capacity


def embed(
        image: Image.Image,
        message: bytes,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> Image.Image:
    if bits_per_channel != 1:
        raise ValueError("bits_per_channel must be 1 in this mode")
    capacity = calculate_capacity(image, 1, channels)

    length = len(message)
    length_bytes = length.to_bytes(4, byteorder="big")
    payload = length_bytes + message

    if len(payload) > capacity:
        raise CapacityError("Message is too large for given image and settings")

    payload_bits = bytes_to_bits(payload)

    stego = image.copy()
    pixels = stego.load()
    width, height = stego.size

    bit_index = 0
    total_bits = len(payload_bits)
    for y in range(height):
        for x in range(0, width - 1, 2):
            if bit_index + 1 >= total_bits:
                return stego

            r0, g0, b0 = pixels[x, y]
            r1, g1, b1 = pixels[x + 1, y]
            values_0 = [r0, g0, b0]
            values_1 = [r1, g1, b1]

            for ch in channels:
                if bit_index + 1 >= total_bits:
                    break

                if ch == "R":
                    idx = 0
                elif ch == "G":
                    idx = 1
                elif ch == "B":
                    idx = 2
                else:
                    continue

                value0 = values_0[idx]
                value1 = values_1[idx]
                bit0 = payload_bits[bit_index]
                bit1 = payload_bits[bit_index + 1]

                value0_lb = value0 & 1

                if value0_lb != bit0:
                    if value0 == 0:
                        value0 = 1
                    elif value0 == 255:
                        value0 = 254
                    else:
                        value0 += choice([-1, +1])

                parity = (value0 + value1) % 2
                if parity != bit1:
                    if value1 == 0:
                        value1 = 1
                    elif value1 == 255:
                        value1 = 254
                    else:
                        value1 += choice([-1, +1])

                bit_index += 2
                values_0[idx] = value0
                values_1[idx] = value1

            pixels[x, y] = tuple(values_0)
            pixels[x + 1, y] = tuple(values_1)

    return stego


def extract(
        image: Image.Image,
        bits_per_channel: int = 1,
        channels: str = "RGB"
) -> bytes:
    if bits_per_channel != 1:
        raise ValueError("bits_per_channel must be 1 in this mode")

    pixels = image.load()
    width, height = image.size

    bits: list[int] = []

    for y in range(height):
        for x in range(0, width - 1, 2):
            r0, g0, b0 = pixels[x, y]
            r1, g1, b1 = pixels[x + 1, y]
            values_0 = [r0, g0, b0]
            values_1 = [r1, g1, b1]

            for ch in channels:
                if ch == "R":
                    idx = 0
                elif ch == "G":
                    idx = 1
                elif ch == "B":
                    idx = 2
                else:
                    continue

                value0 = values_0[idx]
                parity = (values_0[idx] + values_1[idx]) % 2
                bit0 = value0 & 1
                bit1 = parity & 1

                bits.append(bit0)
                bits.append(bit1)

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




