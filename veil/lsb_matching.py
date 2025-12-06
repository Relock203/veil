"""Реализация метода LSB Matching для стеганографии в изображениях.

В отличие от классического LSB, который просто принудительно устанавливает
младший бит пикселя в нужное значение, LSB Matching изменяет значение пикселя
на +1 или -1, если бит не совпадает. Это делает распределение значений
более естественным и усложняет простейший статистический стегоанализ.
"""

from random import choice
from PIL import Image
from core.exceptions import CapacityError, ExtractionError
from core.utils import bytes_to_bits, bits_to_bytes, calculate_capacity


def embed_message(
        image: Image.Image,
        message: bytes,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> Image.Image:
    """Встраивает сообщение методом LSB Matching.

    Метод LSB Matching работает так:
      * если младший бит пикселя уже совпадает с нужным битом сообщения,
        значение пикселя не изменяется;
      * если не совпадает — значение случайно увеличивается или уменьшается
        на единицу (при этом контролируются границы 0 и 255).

    Такое поведение помогает избежать систематического смещения значений,
    характерного для «жёсткого» переписывания LSB, и делает статистику
    яркостей менее подозрительной.

    Формат полезной нагрузки:
      [4 байта длины сообщения в big-endian] + [байты сообщения]

    Ограничение:
      В текущей реализации допускается только bits_per_channel = 1, то есть
      изменяется только один младший бит в каждом канале.

    Args:
        image: Исходное изображение Pillow, в которое встраивается сообщение.
        message: Сообщение для скрытия, в виде байтовой строки.
        bits_per_channel: Количество младших бит, используемых для встраивания.
            Для LSB Matching допустимо только значение 1.
        channels: Строка, задающая используемые цветовые каналы,
            например "R", "G", "B", "RG", "RGB".

    Returns:
        Новое изображение Pillow с внедрённым сообщением.

    Raises:
        ValueError: Если bits_per_channel не равен 1 или указан
            неподдерживаемый канал.
        CapacityError: Если сообщение не помещается в изображение при
            заданных настройках.
    """
    if bits_per_channel != 1:
        raise ValueError("bits_per_channel must be 1 in this mode")

    # Вместимость считаем с учётом одного бита на канал.
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
        for x in range(width):
            r, g, b = pixels[x, y]
            values = [r, g, b]

            # Если все биты полезной нагрузки уже встроены — возвращаем результат.
            if bit_index >= total_bits:
                return stego

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
                    raise ValueError("Unsupported channel")

                value = values[idx]
                last_bit = value & 1
                current_bit = payload_bits[bit_index]

                if last_bit != current_bit:
                    # Меняем яркость на 1 вверх/вниз, не выходя за диапазон [0, 255].
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


def extract_message(
        image: Image.Image,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> bytes:
    """Извлекает сообщение, встроенное методом LSB Matching.

    Извлечение происходит аналогично классическому LSB:
    считываются младшие биты выбранных каналов по всем пикселям изображения
    в том же порядке, в каком происходило встраивание.

    Первые 32 бита интерпретируются как длина сообщения (4 байта, big-endian),
    далее читается указанное количество байт полезной нагрузки.

    Args:
        image: Изображение с предполагаемым стего-сообщением.
        bits_per_channel: Количество младших бит, использованных при встраивании.
            Для LSB Matching должно быть равно 1.
        channels: Набор каналов, использовавшихся при встраивании.

    Returns:
        Извлечённое сообщение в виде байтов.

    Raises:
        ValueError: Если bits_per_channel не равен 1.
        ExtractionError: Если недостаточно бит для чтения длины или всего сообщения.
    """
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
