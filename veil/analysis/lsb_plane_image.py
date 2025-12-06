"""
Генерация визуализации LSB-плоскости (Least Significant Bit Plane).

LSB-плоскость — это изображение, где каждый пиксель показывает младший бит
соответствующего цветового канала исходного изображения. Такое отображение
часто используется в стегоанализе для выявления подозрительных структур,
возникающих из-за примитивных методов LSB-стеганографии.

Если LSB распределены случайно, изображение выглядит как шум.
Если распределение нарушено — появляются текстуры, сетки и другие аномалии.
"""

from PIL import Image


def lsb_plane_image(image: Image.Image, channel: str = "R") -> Image.Image:
    """Строит изображение LSB-плоскости для выбранного цветового канала.

    Каждый пиксель результата равен:
        - 0   (чёрный), если младший бит канала = 0
        - 255 (белый),  если младший бит канала = 1

    Это позволяет визуально анализировать распределение LSB.

    Args:
        image: Входное изображение (будет преобразовано в RGB).
        channel: Цветовой канал: "R", "G" или "B".

    Returns:
        Изображение mode="L" того же размера, где LSB отображён как ч/б картинка.

    Raises:
        ValueError: Если канал указан неверно.
    """
    ch = channel.upper()
    if ch not in ("R", "G", "B"):
        raise ValueError('channel must be one of "R", "G", "B"')

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()

    result = Image.new("L", (width, height))
    out = result.load()

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            if ch == "R":
                v = r
            elif ch == "G":
                v = g
            else:
                v = b

            out[x, y] = 255 if (v & 1) else 0

    return result
