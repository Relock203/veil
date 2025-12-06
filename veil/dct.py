"""
Экспериментальный метод частотной стеганографии на основе DCT (дискретного
косинусного преобразования), по аналогии с JPEG.

Основная идея:
    1. Изображение преобразуется в пространство YCbCr.
    2. Стеганография выполняется в яркостном канале (Y), т.к. он наиболее
       устойчив к искажениям и важен для зрительного восприятия.
    3. Изображение разбивается на блоки 8×8.
    4. Для каждого блока вычисляется DCT.
    5. Несколько AC-коэффициентов заменяются на их значения с изменённым LSB.
    6. Выполняется обратное DCT → сборка изображения → преобразование в RGB.

Метод нестабилен из-за ошибок округления и применения float в DCT/IDCT.
Используется только как экспериментальная демонстрация частотной стеганографии.
"""

from typing import List, Tuple
from math import cos, pi
from PIL import Image
from core.utils import bytes_to_bits, bits_to_bytes
from core.exceptions import CapacityError, ExtractionError

BLOCK_SIZE = 8

# Частоты, в которые мы встраиваем 1 бит
CANDIDATE_POSITIONS: List[Tuple[int, int]] = [
    (2, 3),
    (3, 2),
    (4, 1),
]


def _image_to_luma_blocks(image: Image.Image) -> Tuple[List[List[List[float]]], int, int]:
    """Преобразует изображение в яркостный канал (L) и разбивает на блоки 8×8.

    Для корректности DCT все размеры должны быть кратны 8, поэтому изображение
    дополняется паддингом справа и снизу до ближайшего размера, делящегося на 8.

    Args:
        image: Исходное изображение Pillow.

    Returns:
        blocks: Список блоков 8×8, каждый блок — матрица float.
        width: Исходная ширина изображения.
        height: Исходная высота изображения.
    """
    luma = image.convert("L")
    width, height = luma.size

    padded_width = (width + 7) // 8 * 8
    padded_height = (height + 7) // 8 * 8

    padded = Image.new("L", (padded_width, padded_height))
    padded.paste(luma, (0, 0))

    pixels = padded.load()
    blocks: List[List[List[float]]] = []

    for y in range(0, padded_height, 8):
        for x in range(0, padded_width, 8):
            block = [
                [float(pixels[x + dx, y + dy]) for dx in range(8)]
                for dy in range(8)
            ]
            blocks.append(block)

    return blocks, width, height


def _luma_blocks_to_image(blocks: List[List[List[float]]], width: int, height: int) -> Image.Image:
    """Собирает изображение из блоков яркости (после IDCT).

    Блоки записываются в изображение с учётом паддинга; затем изображение
    обрезается до исходного размера.

    Args:
        blocks: Список блоков 8×8.
        width: Исходная ширина изображения.
        height: Исходная высота изображения.

    Returns:
        Восстановленное изображение в режиме L.
    """
    padded_width = (width + 7) // 8 * 8
    padded_height = (height + 7) // 8 * 8

    img = Image.new("L", (padded_width, padded_height))
    pixels = img.load()

    idx = 0
    for y in range(0, padded_height, 8):
        for x in range(0, padded_width, 8):
            block = blocks[idx]
            idx += 1

            for dy in range(8):
                for dx in range(8):
                    val = round(block[dy][dx])
                    val = max(0, min(255, int(val)))
                    pixels[x + dx, y + dy] = val

    img = img.crop((0, 0, width, height))
    return img


def _dct_2d(block: List[List[float]]) -> List[List[float]]:
    """Вычисляет двумерное дискретное косинусное преобразование (DCT-II).

    Полная реализация формулы DCT-II без квантования.

    Args:
        block: Матрица 8×8 яркостей.

    Returns:
        Матрица коэффициентов DCT 8×8.
    """
    n = BLOCK_SIZE
    result = [[0.0 for _ in range(n)] for _ in range(n)]

    def alpha(k: int) -> float:
        return 1 / (2 ** 0.5) if k == 0 else 1.0

    for v in range(n):
        for u in range(n):
            sum_val = 0.0
            for y in range(n):
                for x in range(n):
                    sum_val += (
                            block[y][x]
                            * cos((2 * x + 1) * u * pi / (2 * n))
                            * cos((2 * y + 1) * v * pi / (2 * n))
                    )

            result[v][u] = 0.25 * alpha(u) * alpha(v) * sum_val

    return result


def _idct_2d(coeffs: List[List[float]]) -> List[List[float]]:
    """Выполняет обратное двумерное DCT (IDCT).

    Args:
        coeffs: Матрица коэффициентов DCT 8×8.

    Returns:
        Восстановленный блок яркости 8×8.
    """
    n = BLOCK_SIZE
    result = [[0.0 for _ in range(n)] for _ in range(n)]

    def alpha(k: int) -> float:
        return 1 / (2 ** 0.5) if k == 0 else 1.0

    for y in range(n):
        for x in range(n):
            sum_val = 0.0
            for v in range(n):
                for u in range(n):
                    sum_val += (
                            alpha(u)
                            * alpha(v)
                            * coeffs[v][u]
                            * cos((2 * x + 1) * u * pi / (2 * n))
                            * cos((2 * y + 1) * v * pi / (2 * n))
                    )

            result[y][x] = 0.25 * sum_val
    return result


def calculate_dct_capacity(image: Image.Image) -> int:
    """Вычисляет вместимость частотного контейнера в байтах.

    Каждому блоку 8×8 соответствует len(CANDIDATE_POSITIONS) бит полезных данных.
    """
    blocks = _image_to_luma_blocks(image)[0]
    num_blocks = len(blocks)
    total_bits = num_blocks * len(CANDIDATE_POSITIONS)
    return total_bits // 8


def embed_message(image: Image.Image, message: bytes) -> Image.Image:
    """Встраивает сообщение методом DCT-LSB в яркостный канал изображения.

    Метод:
        * изображение переводится в YCbCr;
        * яркостный канал (Y) разбивается на блоки 8×8;
        * в несколько AC-коэффициентов каждого блока встраивается по 1 биту;
        * выполняется обратное DCT и сборка изображения.

    Встраивается "тройной заголовок" (length * 3), чтобы повысить шанс
    корректного восстановления длины при погрешностях float.

    Args:
        image: Изображение RGB.
        message: Байты сообщения.

    Returns:
        Изображение RGB со встроенными данными.

    Raises:
        CapacityError: Если данные не помещаются в изображение.
    """
    ycbcr = image.convert("YCbCr")
    Y, Cb, Cr = ycbcr.split()

    capacity = calculate_dct_capacity(image)

    length = len(message)
    header = length.to_bytes(4, byteorder="big")
    header3 = header * 3  # 12 байт, 96 бит

    payload = header3 + message

    if len(payload) > capacity:
        raise CapacityError("Message is too large for DCT capacity of this image")

    payload_bits = bytes_to_bits(payload)

    blocks, width, height = _image_to_luma_blocks(Y)

    bit_index = 0
    total_bits = len(payload_bits)

    new_blocks: List[List[List[float]]] = []

    for block in blocks:
        coeffs = _dct_2d(block)

        if bit_index < total_bits:
            for (v, u) in CANDIDATE_POSITIONS:
                if bit_index >= total_bits:
                    break

                bit = payload_bits[bit_index]

                val = coeffs[v][u]
                ival = int(round(val))
                ival = (ival & ~1) | bit
                coeffs[v][u] = float(ival)

                bit_index += 1

        new_blocks.append(coeffs)

    spatial_blocks = [_idct_2d(c) for c in new_blocks]
    stego_Y = _luma_blocks_to_image(spatial_blocks, width, height)

    stego_ycbcr = Image.merge("YCbCr", (stego_Y, Cb, Cr))
    return stego_ycbcr.convert("RGB")


def extract_message(image: Image.Image) -> bytes:
    """Извлекает сообщение, встроенное методом DCT-LSB.

    Возврат длины:
        Встраивается три заголовка подряд, поэтому при извлечении
        вычисляются три разных значения длины (h1, h2, h3).
        Если хотя бы два совпадают — принимаем это значение.

    Args:
        image: Стего-изображение RGB.

    Returns:
        Извлечённое сообщение.

    Raises:
        ExtractionError: Если не удаётся корректно извлечь длину
            или полезные данные.
    """
    ycbcr = image.convert("YCbCr")
    Y, _, _ = ycbcr.split()

    blocks, width, height = _image_to_luma_blocks(Y)

    bits: list[int] = []

    for block in blocks:
        coeffs = _dct_2d(block)

        for (v, u) in CANDIDATE_POSITIONS:
            val = coeffs[v][u]
            ival = int(round(val))
            bits.append(ival & 1)

    if len(bits) < 96:
        raise ExtractionError("Not enough data to read header")

    header_bits = bits[:96]
    h1_bits = header_bits[0:32]
    h2_bits = header_bits[32:64]
    h3_bits = header_bits[64:96]

    h1 = int.from_bytes(bits_to_bytes(h1_bits), "big")
    h2 = int.from_bytes(bits_to_bytes(h2_bits), "big")
    h3 = int.from_bytes(bits_to_bytes(h3_bits), "big")

    candidates = [h1, h2, h3]
    length = None
    for val in candidates:
        if candidates.count(val) >= 2:
            length = val
            break

    if length is None:
        raise ExtractionError("Cannot reliably decode message length")

    needed_bits = length * 8
    total_needed = 96 + needed_bits

    if len(bits) < total_needed:
        raise ExtractionError("Not enough data to read full message")

    message_bits = bits[96:total_needed]
    return bits_to_bytes(message_bits)
