from typing import Dict, List

from PIL import Image


def _iter_channels(channels: str) -> List[str]:
    """Return normalized list of channels, leaving only R/G/B."""
    result: List[str] = []
    for ch in channels.upper():
        if ch in ("R", "G", "B"):
            result.append(ch)
    return result


def lsb_statistics(image: Image.Image, channels: str = "RGB") -> Dict[str, Dict[str, float]]:
    """Compute basic LSB statistics for given channels.

    Для каждого канала считаем:
      - zeros: количество нулевых LSB
      - ones: количество единичных LSB
      - total: zeros + ones
      - p0: доля нулей
      - p1: доля единиц
      - chi2: простейший χ²-тест против равномерного распределения (0.5 / 0.5)

    Args:
        image: Входное изображение (будет сконвертировано в RGB).
        channels: Строка с каналами, например "R", "RG", "RGB".

    Returns:
        Словарь вида:
        {
          "R": {"zeros": ..., "ones": ..., "total": ..., "p0": ..., "p1": ..., "chi2": ...},
          "G": {...},
          ...
        }
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()

    ch_list = _iter_channels(channels)
    stats: Dict[str, Dict[str, float]] = {
        ch: {"zeros": 0.0, "ones": 0.0, "total": 0.0, "p0": 0.0, "p1": 0.0, "chi2": 0.0}
        for ch in ch_list
    }

    if not ch_list:
        return stats

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            values = {"R": r, "G": g, "B": b}
            for ch in ch_list:
                v = values[ch]
                lsb = v & 1
                if lsb == 0:
                    stats[ch]["zeros"] += 1
                else:
                    stats[ch]["ones"] += 1

    for ch in ch_list:
        zeros = stats[ch]["zeros"]
        ones = stats[ch]["ones"]
        total = zeros + ones
        stats[ch]["total"] = total

        if total > 0:
            p0 = zeros / total
            p1 = ones / total
            stats[ch]["p0"] = p0
            stats[ch]["p1"] = p1

            # простейший χ² для двух категорий при ожидаемом 0.5/0.5
            expected = total / 2.0
            chi2 = 0.0
            if expected > 0:
                chi2 = ((zeros - expected) ** 2) / expected + ((ones - expected) ** 2) / expected
            stats[ch]["chi2"] = chi2
        else:
            stats[ch]["p0"] = 0.0
            stats[ch]["p1"] = 0.0
            stats[ch]["chi2"] = 0.0

    return stats


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
