"""
Реализация метода LSBMR (LSB Matching Revisited) для стеганографии.

LSBMR — усовершенствование метода LSB Matching, использующее пары соседних
пикселей. В одной паре кодируются два бита:

    * бит0 задаётся через младший бит первого пикселя;
    * бит1 задаётся через нечётность (parity) суммы двух пикселей.

Такая схема снижает типичные статистические признаки простой LSB-стеганографии
и делает скрытие информации менее заметным для детекторов на основе χ²
и распределений LSB.
"""

from PIL import Image
from random import choice
from core.exceptions import CapacityError, ExtractionError
from core.utils import bytes_to_bits, bits_to_bytes, calculate_capacity


def embed_message(
        image: Image.Image,
        message: bytes,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> Image.Image:
    """Встраивает сообщение методом LSBMR (LSB Matching Revisited).

    Метод LSBMR работает с парами соседних пикселей (p0, p1). На один канал
    пары кодируется сразу два бита полезной нагрузки:

        - первый бит (bit0) задаётся через младший бит p0;
        - второй бит (bit1) задаётся через (p0 + p1) % 2.

    Если младший бит p0 не совпадает с bit0, пиксель p0 изменяется на ±1
    (при соблюдении диапазона 0–255). Если нечётность суммы (p0 + p1)
    не совпадает с bit1, аналогично корректируется p1.

    Такой подход уменьшает характерные следы классического LSB и LSB Matching.

    Формат полезной нагрузки:
        [4 байта длины сообщения в big-endian] + [байты сообщения]

    Ограничение:
        bits_per_channel должен быть равен 1, поскольку схема основана
        на использовании единичных младших битов.

    Args:
        image: Исходное изображение Pillow.
        message: Встраиваемые данные (bytes).
        bits_per_channel: Количество используемых младших бит. Для LSBMR — только 1.
        channels: Используемые каналы, например "R", "G", "B" или "RGB".

    Returns:
        Изображение с внедрённым сообщением.

    Raises:
        ValueError: Если bits_per_channel != 1 или канал некорректный.
        CapacityError: Если сообщение не помещается в изображение при
            заданных настройках.
    """
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
        for x in range(0, width - 1, 2):  # читаем по парам пикселей
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
                    raise ValueError("Unsupported channel")

                value0 = values_0[idx]
                value1 = values_1[idx]
                bit0 = payload_bits[bit_index]
                bit1 = payload_bits[bit_index + 1]

                if (value0 & 1) != bit0:
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


def extract_message(
        image: Image.Image,
        bits_per_channel: int = 1,
        channels: str = "RGB"
) -> bytes:
    """Извлекает сообщение, встроенное методом LSBMR.

    Сообщение декодируется парами пикселей в том же порядке, что и при встраивании.
    Для каждой пары (p0, p1) извлекаются два бита:

        bit0 = LSB(p0)
        bit1 = (p0 + p1) % 2

    Далее первые 32 бита интерпретируются как длина сообщения, а затем
    извлекаются оставшиеся биты полезной нагрузки.

    Args:
        image: Изображение Pillow с внедрёнными данными.
        bits_per_channel: Должен быть равен 1.
        channels: Набор каналов, использованных при встраивании.

    Returns:
        Извлечённые байты сообщения.

    Raises:
        ValueError: Если bits_per_channel != 1.
        ExtractionError: Если данных недостаточно для чтения длины
            или полного сообщения.
    """
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
                value1 = values_1[idx]

                bit0 = value0 & 1
                bit1 = (value0 + value1) % 2

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
    return bits_to_bytes(message_bits)
