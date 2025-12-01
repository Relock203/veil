from PIL import Image
import core.utils as utils
from core.exceptions import CapacityError, ExtractionError


def embed_message(
        image: Image.Image,
        message: bytes,
        channels: str = "R"
) -> Image.Image:
    capacity = utils.calculate_pvd_capacity(image, channels)

    length = len(message)
    length_bytes = length.to_bytes(4, byteorder='big')
    payload = length_bytes + message

    if len(payload) > capacity:
        raise CapacityError("Message is too large for given image and settings")

    payload_bits = utils.bytes_to_bits(payload)

    stego = image.copy()
    pixels = stego.load()
    width, height = stego.size

    bit_index = 0
    total_bits = len(payload_bits)
    for y in range(height):
        for x in range(0, width - 1, 2):
            if bit_index >= total_bits:
                return stego

            r0, g0, b0 = pixels[x, y]
            r1, g1, b1 = pixels[x + 1, y]

            for ch in channels:
                if ch == "R":
                    v0, v1 = r0, r1
                elif ch == "G":
                    v0, v1 = g0, g1
                elif ch == "B":
                    v0, v1 = b0, b1
                else:
                    raise ValueError("Unsupported channel")

                d = abs(v1 - v0)
                lower, upper, k = utils.get_range_info(d)
                chunk = payload_bits[bit_index:bit_index + k]
                if len(chunk) < k:
                    chunk = chunk + [0] * (k - len(chunk))

                b_val = int(''.join(str(b) for b in chunk), 2)
                d_new = utils.choose_target_difference(d, lower, upper, k, b_val)

                sign = 1 if v1 >= v0 else -1

                if sign == 1:
                    min_v0 = 0
                    max_v0 = 255 - d_new
                else:
                    min_v0 = d_new
                    max_v0 = 255

                v0_new = min(max(v0, min_v0), max_v0)
                v1_new = v0_new + sign * d_new

                v0_new = max(0, min(255, v0_new))
                v1_new = max(0, min(255, v1_new))

                if ch == "R":
                    r0, r1 = v0_new, v1_new
                elif ch == "G":
                    g0, g1 = v0_new, v1_new
                elif ch == "B":
                    b0, b1 = v0_new, v1_new

                bit_index += k

            pixels[x, y] = (r0, g0, b0)
            pixels[x + 1, y] = (r1, g1, b1)

    return stego


def extract_message(
        image: Image.Image,
        channels: str = "R",
) -> bytes:
    pixels = image.load()
    width, height = image.size

    bits: list[int] = []

    for y in range(height):
        for x in range(0, width - 1, 2):
            r0, g0, b0 = pixels[x, y]
            r1, g1, b1 = pixels[x + 1, y]

            for ch in channels:
                if ch == "R":
                    v0, v1 = r0, r1
                elif ch == "G":
                    v0, v1 = g0, g1
                elif ch == "B":
                    v0, v1 = b0, b1
                else:
                    raise ValueError("Unsupported channel")

                d = abs(v1 - v0)
                lower, upper, k = utils.get_range_info(d)
                b_val = d % (2**k)
                for i in reversed(range(k)):  # i = k-1, k-2, ..., 0
                    bit = (b_val >> i) & 1
                    bits.append(bit)

    if len(bits) < 32:
        raise ExtractionError("Not enough data to read message length")

    length_bits = bits[:32]
    length_bytes = utils.bits_to_bytes(length_bits)
    length = int.from_bytes(length_bytes, byteorder="big")

    needed_bits = length * 8
    total_needed = 32 + needed_bits

    if len(bits) < total_needed:
        raise ExtractionError("Not enough data to read full message")

    message_bits = bits[32:total_needed]
    message_bytes = utils.bits_to_bytes(message_bits)
    return message_bytes




