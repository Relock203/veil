from PIL import Image


def lsb_plane_image(image: Image.Image, channel: str = "R") -> Image.Image:
    """Build a visualization of LSB plane for a given channel.

    Создаёт чёрно-белую картинку (mode "L"), где:
      - 0   → LSB = 0 (чёрный)
      - 255 → LSB = 1 (белый)

    Args:
        image: Входное изображение (конвертируется в RGB).
        channel: Один канал: "R", "G" или "B".

    Returns:
        Изображение mode="L" такого же размера, как исходное.
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

            lsb = v & 1
            out[x, y] = 255 if lsb == 1 else 0

    return result