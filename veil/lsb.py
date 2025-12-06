from PIL import Image
from core.exceptions import CapacityError, ExtractionError
from core.utils import bytes_to_bits, bits_to_bytes, calculate_capacity


def embed_message(
        image: Image.Image,
        message: bytes,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> Image.Image:
    """Встраивает сообщение в изображение методом базовой LSB-стеганографии.

        Метод LSB (Least Significant Bit) изменяет младшие биты цветовых каналов
        каждого пикселя изображения. Благодаря тому, что младшие биты вносят
        минимальный вклад в итоговый цвет, изменения визуально практически
        незаметны, что делает метод простым, но эффективным в учебных целях.

        Формат полезной нагрузки::

            [4 байта длины сообщения в big-endian] + [данные сообщения]

        Алгоритм работы:
            1. Вычисляется вместимость изображения с учётом переданных параметров.
            2. Формируется блок полезной нагрузки: длина + данные.
            3. Полезная нагрузка переводится в поток бит.
            4. Биты записываются в младшие разряды выбранных каналов (R/G/B).
               Если bits_per_channel > 1, то в один канал записывается несколько бит сразу.
            5. Процесс продолжается построчно по пикселям, пока все биты
               полезной нагрузки не будут встроены.

        Args:
            image: Объект Pillow Image, используемый как контейнер.
            message: Скрываемое сообщение в виде байтов.
            bits_per_channel: Количество младших бит каждого канала,
                в которые разрешено встраивание (1–8, зависит от задачи).
            channels: Строка из символов «R», «G», «B», определяющая,
                какие каналы используются для встраивания.

        Returns:
            Новое изображение Pillow с внедрённым сообщением.

        Raises:
            CapacityError: Если сообщение (включая длину) не помещается
                в доступные биты изображения.
            ValueError: Если передан неподдерживаемый канал.
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
                    raise ValueError("Unsupported channel")

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


def extract_message(
        image: Image.Image,
        bits_per_channel: int = 1,
        channels: str = "RGB",
) -> bytes:
    """Извлекает сообщение, встроенное методом LSB-стеганографии.

        Ожидается тот же формат полезной нагрузки, который формирует embed_message:

            [4 байта длины сообщения] + [сообщение]

        Алгоритм работы:
            1. Побитово извлекаются младшие биты выбранных цветовых каналов.
            2. Из первых 32 бит формируется длина сообщения.
            3. Далее извлекается указанное количество бит данных.
            4. Биты конвертируются обратно в байты.

        Args:
            image: Изображение, содержащее скрытое сообщение.
            bits_per_channel: Количество бит, использованных при встраивании.
            channels: Каналы, по которым происходило встраивание.

        Returns:
            Извлечённое сообщение в виде байтов.

        Raises:
            ExtractionError: Если данных недостаточно для чтения длины
                или полного сообщения.
            ValueError: Если указан некорректный канал.
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
