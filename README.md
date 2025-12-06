# Veil — многофункциональный инструмент стеганографии и стегоанализа

Veil — это учебно-исследовательский инструмент для скрытия и обнаружения информации в цифровых изображениях.
Проект реализует широкий набор современных методов стеганографии, встроенный стегоанализ,
полный CLI-интерфейс и единый API. Подходит для учебных проектов и исследований.

## Основные возможности

### Стеганография
Реализованы методы:
- LSB
- LSB Matching
- LSBMR
- PVD
- Alpha-LSB
- Border-LSB
- DCT-LSB (экспериментальный)

### Стегоанализ
- Визуализация LSB-плоскостей
- LSB-статистика и χ²-тест

## Установка

```
git clone https://github.com/Relock203/veil.git
cd veil
pip install -r requirements.txt
```

## Использование CLI

### Встраивание
```
python -m veil embed --method lsb --input in.png --out out.png --message "hello"
```

### Извлечение
```
python -m veil extract --method lsb --input out.png
```

### Стегоанализ
```
python -m veil analyze --input img.png --lsb-stats
python -m veil analyze --input img.png --lsb-plane-out plane.png --channel R
```

## Тестирование
```
pytest
```

## Архитектура проекта

```
veil/
├── core/
├── veil/
├── analysis/
├── tests/
└── docs/
```
