import os
import sys
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта, чтобы Sphinx видел veil и core
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

project = "Veil"
author = "Relock (Александр)"
copyright = f"{datetime.now().year}"

# Версию можно поменять на актуальную
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",       # Автодок из docstring
    "sphinx.ext.autosummary",   # Генерация API
    "sphinx.ext.napoleon",      # Google-style docstrings
    "sphinx.ext.viewcode",      # Линки на исходники
    "sphinx.ext.todo",          # TODO в документации
]

autosummary_generate = True
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "ru"
