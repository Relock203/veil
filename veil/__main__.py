"""Точка входа для запуска Veil как модуля.

Пример:
    python -m veil embed --method lsb --input in.png --out out.png --message "hello"
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())