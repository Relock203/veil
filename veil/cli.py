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


# ---- РЕГИСТР МЕТОДОВ ---------------------------------------------------------


class MethodAdapter:
    """Адаптер для разных модулей стеганографии.

    Предполагаем, что у всех есть функции:
      - embed_message(image, message, **kwargs) -> Image.Image
      - extract_message(image, **kwargs) -> bytes

    Если в каком-то модуле у тебя другие имена (например, extract вместо
    extract_message), здесь можно легко подправить.
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
        supports_bits_per_channel=False,  # у нас pvd фиксированный (k зависит от диапазона)
        supports_channels=True,
    ),
    "alpha": MethodAdapter(
        "alpha",
        embed_func=alpha.embed_message,
        extract_func=alpha.extract_message,
        supports_bits_per_channel=True,
        supports_channels=False,  # работаем только по альфа-каналу
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


# ---- УТИЛИТЫ ДЛЯ ВХОД/ВЫХОД ДАННЫХ -------------------------------------------


def _read_message_from_args(args: argparse.Namespace) -> bytes:
    """Получить message из --message / --message-file / stdin."""
    if args.message is not None:
        return args.message.encode("utf-8")

    if args.message_file is not None:
        path = Path(args.message_file)
        return path.read_bytes()

    # если ничего не указано — читаем из stdin
    data = sys.stdin.buffer.read()
    if not data:
        raise ValueError("No message provided (empty stdin)")
    return data


def _write_message_to_output(data: bytes, args: argparse.Namespace) -> None:
    """Записать извлечённое сообщение в файл или stdout."""
    if args.out is None:
        # Пытаемся напечатать как utf-8; если не получается — hex
        try:
            text = data.decode("utf-8")
            print(text)
        except UnicodeDecodeError:
            print(data.hex())
    else:
        Path(args.out).write_bytes(data)


# ---- КОМАНДА EMBED -----------------------------------------------------------


def cmd_embed(args: argparse.Namespace) -> int:
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
    # сохраняем в том формате, который ожидается по расширению
    stego.save(out_path)
    print(f"Message embedded using {method_name}, saved to {out_path}")
    return 0


# ---- КОМАНДА EXTRACT --------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
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


# ---- КОМАНДА ANALYZE --------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    image_path = Path(args.input)
    if not image_path.exists():
        print(f"Input image not found: {image_path}", file=sys.stderr)
        return 1

    image = Image.open(image_path)

    # LSB статистика
    if args.lsb_stats:
        stats = lsb_statistics(image, channels=args.channels)
        print("LSB statistics:")
        for ch, st in stats.items():
            print(
                f"  {ch}: zeros={st['zeros']:.0f}, ones={st['ones']:.0f}, "
                f"p0={st['p0']:.3f}, p1={st['p1']:.3f}, chi2={st['chi2']:.3f}"
            )

    # LSB плоскость
    if args.lsb_plane_out is not None:
        plane = lsb_plane_image(image, channel=args.channel)
        out_path = Path(args.lsb_plane_out)
        plane.save(out_path)
        print(f"LSB plane ({args.channel}) saved to {out_path}")

    if not args.lsb_stats and args.lsb_plane_out is None:
        print("Nothing to analyze: use --lsb-stats and/or --lsb-plane-out", file=sys.stderr)
        return 1

    return 0


# ---- ПАРСЕР АРГУМЕНТОВ ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veil",
        description="Veil: steganography toolkit (educational project).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # embed
    p_embed = subparsers.add_parser("embed", help="Embed a message into an image")
    p_embed.add_argument("--method", "-m", required=True, choices=sorted(METHODS.keys()))
    p_embed.add_argument("--input", "-i", required=True, help="Input image path")
    p_embed.add_argument("--out", "-o", required=True, help="Output stego image path")

    msg_group = p_embed.add_mutually_exclusive_group()
    msg_group.add_argument("--message", help="Message text (UTF-8)")
    msg_group.add_argument("--message-file", help="Path to file with raw bytes to embed")
    # если оба не указаны — читаем из stdin

    p_embed.add_argument(
        "--bits-per-channel",
        type=int,
        help="Bits per channel (depends on method)",
    )
    p_embed.add_argument(
        "--channels",
        help='Channels to use, e.g. "R", "RG", "RGB" (depends on method)',
    )
    p_embed.set_defaults(func=cmd_embed)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract a message from an image")
    p_extract.add_argument("--method", "-m", required=True, choices=sorted(METHODS.keys()))
    p_extract.add_argument("--input", "-i", required=True, help="Input stego image path")
    p_extract.add_argument(
        "--out",
        "-o",
        help="Output file for extracted data (if omitted, prints to stdout)",
    )
    p_extract.add_argument(
        "--bits-per-channel",
        type=int,
        help="Bits per channel (must match embed settings)",
    )
    p_extract.add_argument(
        "--channels",
        help="Channels used during embedding (must match embed settings)",
    )
    p_extract.set_defaults(func=cmd_extract)

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Run simple steganalysis (LSB statistics, LSB-plane).",
    )
    p_analyze.add_argument("--input", "-i", required=True, help="Input image path")
    p_analyze.add_argument(
        "--lsb-stats",
        action="store_true",
        help="Print LSB statistics for given channels",
    )
    p_analyze.add_argument(
        "--channels",
        default="RGB",
        help='Channels for LSB statistics, e.g. "R", "RG", "RGB". Default: RGB',
    )
    p_analyze.add_argument(
        "--lsb-plane-out",
        help="If set, save LSB plane image for given channel to this path",
    )
    p_analyze.add_argument(
        "--channel",
        default="R",
        help='Channel for LSB-plane image ("R", "G", or "B"), default: R',
    )
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
