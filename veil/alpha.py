"""
Стеганография в альфа-канале изображения.

Метод использует прозрачность (A-канал) для встраивания данных в формате LSB.
Так как альфа-канал обычно слабо влияет на визуальное восприятие изображения,
встраивание даже нескольких младших бит в каждый пиксель остаётся малозаметным.

Поддерживается от 1 до 8 младших бит на канал, что даёт высокую вместимость
и делает метод удобным для учебных демонстраций и переноса больших сообщений.
"""

from __future__ import annotations
from typing import List
from PIL import Image
from core.utils import bytes_to_bits, bits_to_bytes, calculate_alpha_capacity
from core.exceptions import CapacityError, ExtractionError


def embed_message(
        image: Image.Image,
        message: bytes,
        bits_per_channel: int = 1,
) -> Image.Image:
    """Встраивает сообщение в альфа-канал изображения.

    Метод использует технику LSB (Least Significant Bits) для скрытия данных
    в прозрачности (A-канале). Количество используемых младших бит регулируется
    параметром bits_per_channel (диапазон 1–8).

    Формат полезной нагрузки::
        [4 байта длины сообщения, big-endian] + [байты сообщения]

    Алгоритм:
        1. Исходное изображение переводится в режим RGBA.
        2. Проверяется вместимость альфа-канала.
        3. Сообщение переводится в поток бит.
        4. В каждый пиксель записываются bits_per_channel младших бит.

    Args:
        image: Исходное изображение (RGB или RGBA).
        message: Скрываемые данные в виде байтов.
        bits_per_channel: Количество младших бит для встраивания (1..8).

    Returns:
        Новое изображение RGBA с внедрённым сообщением.

    Raises:
        ValueError: Если bits_per_channel вне диапазона [1, 8].
        CapacityError: Если сообщение не помещается при данных настройках.
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
    mask = (1 << bits_per_channel) - 1  # маска младших бит

    for y in range(height):
        for x in range(width):
            if bit_index >= total_bits:
                return rgba

            r, g, b, a = pixels[x, y]

            remaining = total_bits - bit_index
            if remaining >= bits_per_channel:
                chunk = payload_bits[bit_index: bit_index + bits_per_channel]
            else:
                # Дополняем нулями, если данные закончились
                chunk = payload_bits[bit_index:total_bits] + [0] * (
                        bits_per_channel - remaining
                )

            # Превращаем битовую последовательность в число
            value = int("".join(str(bv) for bv in chunk), 2)

            new_a = (a & ~mask) | value

            pixels[x, y] = (r, g, b, new_a)
            bit_index += bits_per_channel

    return rgba


def extract_message(
        image: Image.Image,
        bits_per_channel: int = 1,
) -> bytes:
    """Извлекает сообщение, скрытое в альфа-канале.

    Извлечение выполняется в том же порядке и объёме бит, что и при встраивании.
    Первые 32 бита интерпретируются как длина сообщения, затем читаются данные.

    Args:
        image: Изображение (RGB или RGBA) с возможным стего-сообщением.
        bits_per_channel: Количество бит, использованное при встраивании (1..8).

    Returns:
        Извлечённое сообщение в виде bytes.

    Raises:
        ValueError: Если bits_per_channel вне диапазона 1..8.
        ExtractionError: Если данных недостаточно для корректного извлечения.
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

            # Получаем строку бит фиксированной длины bits_per_channel
            bitstring = format(value, f"0{bits_per_channel}b")
            bits.extend(int(b) for b in bitstring)

    if len(bits) < 32:
        raise ExtractionError("Not enough data to read message length")

    # Длина сообщения — первые 32 бита
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
