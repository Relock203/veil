"""
Командный интерфейс (CLI) для стеганографического инструмента Veil.

CLI предоставляет три основные группы команд:

1) veil embed   — встроить сообщение в изображение.
2) veil extract — извлечь сообщение из изображения.
3) veil analyze — выполнить базовый стегоанализ (статистика LSB, LSB-плоскости).

Каждый метод стеганографии (LSB, LSB Matching, LSBMR, PVD, Alpha, Border-LSB, DCT)
описан в виде адаптера MethodAdapter. Это позволяет единообразно вызывать
embed_message() и extract_message() независимо от того, какой конкретный
алгоритм выбран пользователем.

CLI поддерживает:
    • выбор метода стеганографии;
    • встраивание текста или бинарных файлов;
    • чтение сообщения из stdin;
    • вывод извлечённого сообщения в stdout или в файл;
    • выбор каналов (если метод поддерживает);
    • регулировку числа используемых младших бит (если метод поддерживает);
    • простые инструменты стегоанализа.

Файл является ключевой точкой входа при вызове "python -m veil".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from PIL import Image

from core.exceptions import CapacityError, ExtractionError

from veil import lsb, lsb_matching, lsbmr, pvd, alpha, border_lsb, dct
from veil.analysis.lsb_plane_image import lsb_plane_image
from veil.analysis.lsb_statistics import lsb_statistics


# ============================================================================
#  Адаптер методов
# ============================================================================


class MethodAdapter:
    """Адаптер для разных модулей стеганографии.

    Все методы стеганографии должны предоставлять две функции:
        embed_message(image, message, **kwargs) -> Image.Image
        extract_message(image, **kwargs) -> bytes

    Параметры:
        name: Имя метода (используется в CLI).
        embed_func: Функция встраивания.
        extract_func: Функция извлечения.
        supports_bits_per_channel: Может ли метод управлять числом младших бит.
        supports_channels: Поддерживает ли метод выбор каналов.

    Это позволяет добавлять новые методы в проект без изменения логики CLI.
    """

    def __init__(
        self,
        name: str,
        embed_func: Callable[..., Image.Image],
        extract_func: Callable[..., bytes],
        supports_bits_per_channel: bool = True,
        supports_channels: bool = True,
    ) -> None:
        self.name = name
        self.embed_func = embed_func
        self.extract_func = extract_func
        self.supports_bits_per_channel = supports_bits_per_channel
        self.supports_channels = supports_channels


# ============================================================================
#  Регистрация методов
# ============================================================================

METHODS: Dict[str, MethodAdapter] = {
    "lsb": MethodAdapter(
        "lsb",
        embed_func=lsb.embed_message,
        extract_func=lsb.extract_message,
        supports_bits_per_channel=True,
        supports_channels=True,
    ),
    "lsb-matching": MethodAdapter(
        "lsb-matching",
        embed_func=lsb_matching.embed_message,
        extract_func=lsb_matching.extract_message,
    ),
    "lsbmr": MethodAdapter(
        "lsbmr",
        embed_func=lsbmr.embed_message,
        extract_func=lsbmr.extract_message,
    ),
    "pvd": MethodAdapter(
        "pvd",
        embed_func=pvd.embed_message,
        extract_func=pvd.extract_message,
        supports_bits_per_channel=False,
        supports_channels=True,
    ),
    "alpha": MethodAdapter(
        "alpha",
        embed_func=alpha.embed_message,
        extract_func=alpha.extract_message,
        supports_bits_per_channel=True,
        supports_channels=False,
    ),
    "border-lsb": MethodAdapter(
        "border-lsb",
        embed_func=border_lsb.embed_message,
        extract_func=border_lsb.extract_message,
        supports_bits_per_channel=True,
        supports_channels=True,
    ),
    "dct": MethodAdapter(
        "dct",
        embed_func=dct.embed_message,
        extract_func=dct.extract_message,
        supports_bits_per_channel=False,
        supports_channels=False,
    ),
}


# ============================================================================
#  Вспомогательные функции для чтения/записи сообщений
# ============================================================================


def _read_message_from_args(args: argparse.Namespace) -> bytes:
    """Возвращает сообщение для внедрения (bytes).

    Источники в порядке приоритета:
        1) --message      — текст UTF-8;
        2) --message-file — произвольный бинарный файл;
        3) stdin          — поток ввода, если оба предыдущих аргумента отсутствуют.

    Raises:
        ValueError: Если stdin пуст и сообщение не предоставлено.
    """
    if args.message is not None:
        return args.message.encode("utf-8")

    if args.message_file is not None:
        path = Path(args.message_file)
        return path.read_bytes()

    data = sys.stdin.buffer.read()
    if not data:
        raise ValueError("No message provided (stdin is empty)")

    return data


def _write_message_to_output(data: bytes, args: argparse.Namespace) -> None:
    """Записывает извлечённое сообщение в файл или выводит его в stdout.

    Если декодирование в UTF-8 невозможно, данные выводятся в виде hex-последовательности.
    """
    if args.out is None:
        try:
            print(data.decode("utf-8"))
        except UnicodeDecodeError:
            print(data.hex())
    else:
        Path(args.out).write_bytes(data)


# ============================================================================
#  Команда EMBED
# ============================================================================


def cmd_embed(args: argparse.Namespace) -> int:
    """Команда CLI: встраивание сообщения в изображение.

    Логика:
        • загрузить исходное изображение;
        • получить полезную нагрузку;
        • вызвать embed_message() выбранного метода;
        • сохранить стегоизображение.

    CLI корректно обрабатывает исключения CapacityError и любые другие ошибки.

    Returns:
        Код завершения процесса:
            0 — успех;
            1 — ошибка.
    """
    method_name = args.method
    if method_name not in METHODS:
        print(f"Unknown method: {method_name}", file=sys.stderr)
        return 1

    adapter = METHODS[method_name]
    image_path = Path(args.input)

    if not image_path.exists():
        print(f"Input image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        message = _read_message_from_args(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    image = Image.open(image_path)

    kwargs: Dict[str, Any] = {}
    if adapter.supports_bits_per_channel and args.bits_per_channel is not None:
        kwargs["bits_per_channel"] = args.bits_per_channel
    if adapter.supports_channels and args.channels is not None:
        kwargs["channels"] = args.channels

    try:
        stego = adapter.embed_func(image=image, message=message, **kwargs)
    except CapacityError as e:
        print(f"Capacity error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Embed error: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    stego.save(out_path)
    print(f"Message embedded using {method_name}, saved to {out_path}")
    return 0


# ============================================================================
#  Команда EXTRACT
# ============================================================================


def cmd_extract(args: argparse.Namespace) -> int:
    """Команда CLI: извлечение сообщения из изображения.

    Вызов цепочки:
        • загрузка изображения,
        • вызов extract_message() выбранного метода,
        • вывод результата в stdout или файл.

    CLI корректно обрабатывает:
        • ExtractionError — ошибки формата стего-сообщения;
        • любые другие исключения.

    Returns:
        Код завершения: 0 — успех, 1 — ошибка.
    """
    method_name = args.method
    if method_name not in METHODS:
        print(f"Unknown method: {method_name}", file=sys.stderr)
        return 1

    adapter = METHODS[method_name]
    image_path = Path(args.input)

    if not image_path.exists():
        print(f"Input image not found: {image_path}", file=sys.stderr)
        return 1

    image = Image.open(image_path)

    kwargs: Dict[str, Any] = {}
    if adapter.supports_bits_per_channel and args.bits_per_channel is not None:
        kwargs["bits_per_channel"] = args.bits_per_channel
    if adapter.supports_channels and args.channels is not None:
        kwargs["channels"] = args.channels

    try:
        data = adapter.extract_func(image=image, **kwargs)
    except ExtractionError as e:
        print(f"Extraction error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Extract error: {e}", file=sys.stderr)
        return 1

    _write_message_to_output(data, args)
    return 0


# ============================================================================
#  Команда ANALYZE
# ============================================================================


def cmd_analyze(args: argparse.Namespace) -> int:
    """Команда CLI: базовый стегоанализ.

    Поддерживаются:
        • вывод статистики LSB по каналам (--lsb-stats);
        • построение LSB-плоскости (--lsb-plane-out).

    Если пользователь не указал ни одного из двух параметров, команда сообщает
    об отсутствии действий.

    Returns:
        Код завершения.
    """
    image_path = Path(args.input)

    if not image_path.exists():
        print(f"Input image not found: {image_path}", file=sys.stderr)
        return 1

    image = Image.open(image_path)

    # статистика LSB
    if args.lsb_stats:
        stats = lsb_statistics(image, channels=args.channels)
        print("LSB statistics:")
        for ch, st in stats.items():
            print(
                f"  {ch}: zeros={st['zeros']:.0f}, ones={st['ones']:.0f}, "
                f"p0={st['p0']:.3f}, p1={st['p1']:.3f}, chi2={st['chi2']:.3f}"
            )

    # LSB-плоскость
    if args.lsb_plane_out is not None:
        plane = lsb_plane_image(image, channel=args.channel)
        out_path = Path(args.lsb_plane_out)
        plane.save(out_path)
        print(f"LSB plane ({args.channel}) saved to {out_path}")

    if not args.lsb_stats and args.lsb_plane_out is None:
        print("Nothing to analyze: use --lsb-stats and/or --lsb-plane-out", file=sys.stderr)
        return 1

    return 0


# ============================================================================
#  Построение аргумент-парсера
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Создаёт объект ArgumentParser с полной структурой команд Veil.

    Команды:
        * embed   — скрытие сообщения;
        * extract — извлечение сообщения;
        * analyze — инструменты стегоанализа.

    Возвращает:
        Готовый к работе ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="veil",
        description="Veil: универсальный учебный инструмент стеганографии.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------- EMBED -------------------------
    p_embed = subparsers.add_parser("embed", help="Embed a message into an image")
    p_embed.add_argument("--method", "-m", required=True, choices=sorted(METHODS.keys()))
    p_embed.add_argument("--input", "-i", required=True, help="Input image path")
    p_embed.add_argument("--out", "-o", required=True, help="Output stego image path")

    msg_group = p_embed.add_mutually_exclusive_group()
    msg_group.add_argument("--message", help="Message text (UTF-8)")
    msg_group.add_argument("--message-file", help="Path to file with raw bytes to embed")
    # если ничего не указано — читаем из stdin

    p_embed.add_argument("--bits-per-channel", type=int)
    p_embed.add_argument("--channels", help='Channels, e.g. "R", "RG", "RGB"')
    p_embed.set_defaults(func=cmd_embed)

    # ------------------------- EXTRACT -------------------------
    p_extract = subparsers.add_parser("extract", help="Extract a message from an image")
    p_extract.add_argument("--method", "-m", required=True, choices=sorted(METHODS.keys()))
    p_extract.add_argument("--input", "-i", required=True)
    p_extract.add_argument("--out", "-o", help="Output file (optional)")
    p_extract.add_argument("--bits-per-channel", type=int)
    p_extract.add_argument("--channels")
    p_extract.set_defaults(func=cmd_extract)

    # ------------------------- ANALYZE -------------------------
    p_analyze = subparsers.add_parser(
        "analyze", help="Run simple steganalysis (LSB statistics, LSB-plane)."
    )
    p_analyze.add_argument("--input", "-i", required=True)
    p_analyze.add_argument("--lsb-stats", action="store_true")
    p_analyze.add_argument("--channels", default="RGB")
    p_analyze.add_argument("--lsb-plane-out")
    p_analyze.add_argument("--channel", default="R")
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа в приложение командной строки Veil."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "embed":
            return cmd_embed(args)
        elif args.command == "extract":
            return cmd_extract(args)
        elif args.command == "analyze":
            return cmd_analyze(args)
        else:
            parser.print_help()
            return 1

    except CapacityError as e:
        print(f"[Ошибка вместимости] {e}", file=sys.stderr)
        return 1
    except ExtractionError as e:
        print(f"[Ошибка извлечения] {e}", file=sys.stderr)
        return 1
    except VeilError as e:
        # Любые другие твои кастомные ошибки
        print(f"[Ошибка Veil] {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"[Файл не найден] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[Ошибка параметров] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # На всякий пожарный — чтобы не вывалился traceback
        print(f"[Неожиданная ошибка] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
