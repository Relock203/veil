Использование Veil
==================

Запуск CLI
----------

Veil можно запускать как модуль:

.. code-block:: console

    python -m veil --help

Встраивание текстового сообщения:

.. code-block:: console

    python -m veil embed \\
        --method lsb \\
        --input image.png \\
        --out stego.png \\
        --message "hello"

Извлечение сообщения:

.. code-block:: console

    python -m veil extract \\
        --method lsb \\
        --input stego.png

Использование Python API
------------------------

.. code-block:: python

    from veil.lsb import embed_message, extract_message
    from PIL import Image

    img = Image.open("photo.png")
    stego = embed_message(img, b"secret")
    secret = extract_message(stego)
