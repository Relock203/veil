from pathlib import Path
from typing import Optional
from PIL import Image
from core.exceptions import CapacityError, ExtractionError
from core.image_io import load_image, save_image
from core.utils import bytes_to_bits, bits_to_bytes


def calculate_capacity(image: Image.Image, bits_per_channel: int, channels: str) -> int:
    """Calculate maximum payload capacity for the given image and settings.

    Args:
        image: Cover image.
        bits_per_channel: Number of low-order bits per color channel used
            for embedding (1–3 recommended).
        channels: Color channels to use, e.g. "RGB", "RG", "B".

    Returns:
        Maximum number of bytes that can be embedded into the image.
    """
    if bits_per_channel <= 0:
        raise ValueError("bits_per_channel must be positive")

    width, height = image.size
    num_pixels = width * height
    num_channels = len(channels)
    total_bits = num_pixels * bits_per_channel * num_channels
    return total_bits // 8


def embed_message(
        image: Image.Image,
        message: bytes,
        key: Optional[str] = None,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> Image.Image:
    """Embed a secret message into an image using basic LSB steganography.

    Args:
        image: Pillow Image object to be used as cover image.
        message: Raw bytes of the message to hide.
        key: Optional secret key (currently unused for ``mode="lsb"``).
        bits_per_channel: Number of low-order bits used per color channel.
        channels: Color channels to use, e.g. "RGB", "RG", "B".

    Returns:
        New Pillow Image object containing the embedded message.

    Raises:
        CapacityError: If the message does not fit into the image.
    """

    capacity = calculate_capacity(image, bits_per_channel, channels)

    length = len(message)
    length_bytes = length.to_bytes(length=4, byteorder="big")
    payload = length_bytes + message  # 4 bytes of length + data

    if len(payload) > capacity:
        raise CapacityError("Message is too large for given image and settings")

    payload_bits = bytes_to_bits(payload)

    stego = image.copy()
    pixels = stego.load()
    width, height = stego.size

    bit_index = 0
    total_bits = len(payload_bits)
    mask = (1 << bits_per_channel) - 1

    for y in range(height):
        for x in range(width):
            if bit_index >= total_bits:
                return stego

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
                    chunk = payload_bits[bit_index: bit_index + bits_per_channel]
                else:
                    # padding the tail with zeros
                    chunk = payload_bits[bit_index:total_bits] + [0] * (
                            bits_per_channel - remaining
                    )

                bit_value = int("".join(str(b) for b in chunk), 2)

                value = values[idx]
                new_value = (value & ~mask) | bit_value
                values[idx] = new_value

                bit_index += bits_per_channel

            pixels[x, y] = tuple(values)

    return stego


def extract(
        image: Image.Image,
        key: Optional[str] = None,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> bytes:
    """Extract hidden message from an image encoded by ``embed_message``.

    Args:
        image: Pillow Image object that contains hidden data.
        key: Optional secret key (currently unused for ``mode="lsb"``).
        bits_per_channel: Number of low-order bits per color channel used
            during embedding.
        channels: Color channels used for embedding.

    Returns:
        Raw bytes of the extracted hidden message.

    Raises:
        ExtractionError: If data cannot be correctly extracted.
    """
    pixels = image.load()
    width, height = image.size

    mask = (1 << bits_per_channel) - 1
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
                chunk_value = value & mask

                # converting a number into a sequence of binary digits
                for shift in reversed(range(bits_per_channel)):
                    bit = (chunk_value >> shift) & 1
                    bits.append(bit)

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


def embed_file_to_file(
        input_path: Path,
        output_path: Path,
        message: bytes,
        key: Optional[str] = None,
        mode: str = "lsb",
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> None:
    """Embed message bytes into an image file and save result to another file."""
    image = load_image(input_path)
    stego = embed_message(
        image,
        message=message,
        key=key,
        bits_per_channel=bits_per_channel,
        channels=channels,
    )
    save_image(stego, output_path)


def extract_from_file(
        input_path: Path,
        key: Optional[str] = None,
        mode: str = "lsb",
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> bytes:
    """Extract hidden message bytes from an image file."""
    image = load_image(input_path)
    return extract(
        image,
        key=key,
        bits_per_channel=bits_per_channel,
        channels=channels,
    )
