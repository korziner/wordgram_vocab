# wordgram_vocab
Частотный словарь словосочетаний изъ JSONL-корпуса

```
wordgram_vocab -h
Извлеченіе частотнаго словаря словесныхъ n-граммъ изъ JSONL-корпуса

Usage: wordgram_vocab [OPTIONS] --input <INPUT> --output <OUTPUT>

Options:
  -i, --input <INPUT>
          Путь къ JSONL-файлу (обычный или .zst)
  -o, --output <OUTPUT>
          Путь къ выходному TSV (словосочетаніе\tчастота)
      --word-ngrams <WORD_NGRAMS>
          Размѣры словесныхъ n-граммъ чрезъ запятую (2=пары, 3=тройки, 4=четвёрки) [default: 2,3]
      --separator <SEPARATOR>
          Раздѣлитель словъ въ выходномъ токене [default: _]
      --min-count <MIN_COUNT>
          Минимальная частота для включенія въ словарь [default: 2]
      --max-vocab <MAX_VOCAB>
          Максимальный размѣръ словаря (0 = безъ ограниченіи) [default: 0]
      --min-word-len <MIN_WORD_LEN>
          Минимальная длина слова (0 = не фильтровать) [default: 0]
      --lowercase
          Приводить къ нижнему регистру
      --expected-unique <EXPECTED_UNIQUE>
          Ожидаемое число уникальныхъ n-граммъ (для bloom filter) [default: 20000000]
      --fp-rate <FP_RATE>
          Вѣроятность ложнаго срабатыванія bloom filter [default: 0.01]
      --field <FIELD>
          JSON-поле съ текстомъ [default: text]
      --threads <THREADS>
          Число потоковъ rayon (0 = авто) [default: 0]
      --chunk-size <CHUNK_SIZE>
          Размеръ пачки строкъ [default: 10000]
  -v, --verbose
  ```
