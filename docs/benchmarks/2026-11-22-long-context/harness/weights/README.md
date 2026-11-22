# Контрольный GGUF для замеров длинного контекста

Velri и llama.cpp загружают один синтетический GGUF с 75 тензорами F32. В нём 62 927 616 параметров (251 710 464 байта весов), полный файл занимает 251 943 872 байта. Контрольный SHA-256 — `e0e76115155be432eb30ab5e623d8acc9b4ab6b97f47ad2cda3eb2ea19ed8453`.

## Генерация

Запустите команды в клоне Velri на Linux x86-64 с Python 3.11 или новее, Clang с поддержкой C23, CMake, Ninja и `slangc` в PATH. Переменная `WORK` задаёт каталог сборок, окружения Python и модели. Если CPU-сборка стенда уже существует, укажите её в `VELRI_CPU_BUILD`. Модель и манифест появятся рядом. [Зависимости Python](requirements.txt) · [Запуск измерений](../README.md)

```sh
repo=$(git rev-parse --show-toplevel)
weights="$repo/docs/benchmarks/2026-11-22-long-context/harness/weights"
WORK=${WORK:-"$HOME/.cache/velri-long-context"}
VELRI_CPU_BUILD=${VELRI_CPU_BUILD:-"$WORK/velri-cpu"}
MODEL=${MODEL:-"$WORK/velri-llama-f32.gguf"}
export CC=${CC:-clang}

cmake -S "$repo" -B "$VELRI_CPU_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON -DVELRI_ENABLE_VULKAN=OFF
cmake --build "$VELRI_CPU_BUILD" --target test_engine_runtime

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install -r "$weights/requirements.txt"
"$WORK/venv/bin/python" "$weights/export.py" \
    --velri "$repo" --build "$VELRI_CPU_BUILD" --output "$MODEL"
```

Генератор создаёт модель с исходным порядком RoPE `VELRI_ROTARY_HALF` и вызывает `populate_llama_weights` из тестов движка. Дескриптор задаёт 8 слоёв, ширину 768, 12 голов Q и 4 головы KV размерности 64, SwiGLU шириной 2048 и независимую выходную проекцию. Эпсилон RMSNorm равен 1e−6, база RoPE — 10 000. В GGUF указан контекст 9216 с отключённым масштабированием. [Дескриптор модели](dump.c) · [Генератор весов](../../../../../tests/test_engine_runtime.c)

## Порядок тензоров и проверка

Для 16 матриц Q и K экспортёр переставляет выходные строки каждой головы в порядке `0, 32, 1, 33, …, 31, 63`. Эти строки образуют соседние пары RoPE, которые ожидают llama.cpp и оба стенда Velri при загрузке GGUF. Остальные 59 тензоров записываются побайтно. GGUF содержит синтетический словарь SentencePiece на 8192 токена с байтовым резервом. Оба движка получают одинаковые числовые ID. Автоматическая вставка BOS и EOS отключена. [Экспортёр](export.py) · [Стенд Velri](../velri-gpu.c)

Экспортёр повторно читает GGUF, проверяет типы, формы и SHA-256 каждого тензора, обращает перестановку Q/K и сверяет восстановленные байты с весами модели. Перед сохранением он проверяет SHA-256 всего файла. В `*.manifest.json` записываются хеши каждого тензора в обоих порядках, хеши исходников генератора, ревизия Velri и версия компилятора. [Проверка файла](export.py)
