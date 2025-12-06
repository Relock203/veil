"""
Реализация метода PVD (Pixel Value Differencing) для стеганографии.

Метод PVD (разностная стеганография) использует разность яркостей двух
соседних пикселей в изображении для определения количества бит, которые
можно встроить в данную область. Чем больше локальный контраст (разность
значений), тем большее число бит можно скрыть без заметных артефактов.

Алгоритм адаптивный: изображение автоматически подбирает допустимое
количество скрываемой информации на основе локальных характеристик.
"""

from PIL import Image
import core.utils as utils
from core.exceptions import CapacityError, ExtractionError


def embed_message(
        image: Image.Image,
        message: bytes,
        channels: str = "R"
) -> Image.Image:
    """Встраивает сообщение методом PVD (Pixel Value Differencing).

    Метод работает с парами соседних пикселей: (p0, p1).
    Для каждой пары вычисляется разность d = |p1 - p0|, затем на основании
    величины d выбирается диапазон (lower, upper) и количество бит k, которые
    можно безопасно встроить в данную локальную область.

    Далее:
      * из бит сообщения берутся k бит;
      * эти k бит интерпретируются как число b_val;
      * выбирается новая разность d_new в диапазоне [lower, upper] так,
        чтобы d_new % 2^k == b_val;
      * пиксели корректируются так, чтобы их разность стала именно d_new.

    Такой подход делает метод визуально устойчивым, поскольку большое
    количество данных встраивается преимущественно в области с реальным
    высоким контрастом (границы, текстуры), где искажения слабо заметны.

    Формат полезной нагрузки:
        [4 байта длины в big-endian] + [байты сообщения]

    Args:
        image: Изображение Pillow (RGB).
        message: Данные для скрытия (bytes).
        channels: Строка каналов ("R", "G", "B" или комбинации):
            PVD применяется по каждому выбранному каналу.

    Returns:
        Новое изображение Pillow с внедрёнными данными.

    Raises:
        CapacityError: Если сообщение превышает доступную вместимость метода.
        ValueError: Если указан неподдерживаемый канал.
    """
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
            # Все биты полезной нагрузки встроены — возвращаем изображение.
            if bit_index >= total_bits:
                return stego

            # Берём пару пикселей
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

                # 1) Вычисление существующей разности
                d = abs(v1 - v0)

                # 2) Определяем диапазон (lower, upper) и количество бит k
                lower, upper, k = utils.get_range_info(d)

                # 3) Берём k бит сообщения
                chunk = payload_bits[bit_index:bit_index + k]
                if len(chunk) < k:
                    chunk += [0] * (k - len(chunk))

                b_val = int(''.join(str(b) for b in chunk), 2)

                # 4) Выбираем новую разность d_new, удовлетворяющую условию
                #    d_new % (2^k) == b_val
                d_new = utils.choose_target_difference(d, lower, upper, k, b_val)

                # Определяем знак (порядок пикселей сохраняется)
                sign = 1 if v1 >= v0 else -1

                # Корректируем значения так, чтобы разность стала d_new
                if sign == 1:
                    min_v0 = 0
                    max_v0 = 255 - d_new
                else:
                    min_v0 = d_new
                    max_v0 = 255

                v0_new = min(max(v0, min_v0), max_v0)
                v1_new = v0_new + sign * d_new

                # Ограничиваем результат диапазоном 0–255
                v0_new = max(0, min(255, v0_new))
                v1_new = max(0, min(255, v1_new))

                # Обновляем соответствующий канал
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
    """Извлекает сообщение, встроенное методом PVD.

    Для каждой пары пикселей (v0, v1):
        * вычисляется разность d = |v1 - v0|;
        * определяется число бит k для данной разности;
        * восстанавливается значение b_val = d % (2^k);
        * эти k бит добавляются к общему потоку полезной нагрузки.

    После накопления бит:
        * первые 32 бита интерпретируются как длина сообщения;
        * затем извлекаются оставшиеся биты данных.

    Args:
        image: Изображение с предполагаемыми встроенными данными.
        channels: Каналы, по которым происходило встраивание.

    Returns:
        Извлечённые данные (bytes).

    Raises:
        ExtractionError: Если данных недостаточно для чтения длины или сообщения.
        ValueError: Если канал указан неверно.
    """
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

                # Разбиваем b_val на k бит: старший → младший
                for i in reversed(range(k)):
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
    return utils.bits_to_bytes(message_bits)
