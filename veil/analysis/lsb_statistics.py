"""
Базовый стегоанализ для методов LSB: статистика распределения младших бит.

Модуль вычисляет:
    * количество LSB, равных 0 и 1;
    * долю нулей (p0) и единиц (p1);
    * простейший χ²-критерий против гипотезы равномерного распределения.

Отклонения распределения LSB от случайного часто указывают на применение
примитивных методов LSB-стеганографии.
"""

from typing import Dict, List
from PIL import Image


def _iter_channels(channels: str) -> List[str]:
    """Преобразует строку каналов в список ['R', 'G', 'B'].

    Фильтрует любые неподдерживаемые символы. Например:
        "RGBxyz" → ["R", "G", "B"].
    """
    result: List[str] = []
    for ch in channels.upper():
        if ch in ("R", "G", "B"):
            result.append(ch)
    return result


def lsb_statistics(image: Image.Image, channels: str = "RGB") -> Dict[str, Dict[str, float]]:
    """Вычисляет статистику LSB по выбранным каналам изображения.

    Для каждого канала собирается:
        zeros — число LSB = 0
        ones  — число LSB = 1
        total — общее количество пикселей (zeros + ones)
        p0    — доля нулевых LSB
        p1    — доля единичных LSB
        chi2  — простейший χ²-тест сравнения с равным распределением (0.5 / 0.5)

    χ²-тест:
        chi2 = ( (zeros - exp)^2 / exp ) + ( (ones - exp)^2 / exp ),
        где exp = total / 2.

    Args:
        image: Входное изображение Pillow.
        channels: Каналы для анализа, например: "R", "G", "B", "RG", "RGB".

    Returns:
        Словарь вида::
            {
                "R": {"zeros": ..., "ones": ..., "total": ..., "p0": ..., "p1": ..., "chi2": ...},
                "G": {...},
                "B": {...}
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
                if (v & 1) == 0:
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

            expected = total / 2.0
            if expected > 0:
                chi2 = ((zeros - expected) ** 2) / expected + ((ones - expected) ** 2) / expected
            else:
                chi2 = 0.0

            stats[ch]["chi2"] = chi2
        else:
            stats[ch]["p0"] = 0.0
            stats[ch]["p1"] = 0.0
            stats[ch]["chi2"] = 0.0

    return stats
