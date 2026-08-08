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

Двух слов связать не может в начале претрейна LLM почему?

Эмбеддинги расширяем словосочетанием:
```
time python ./experiments/wordgram_run1/add_wordgrams.py --base_tokenizer /huggingface/hub/models--zakarth--violet-1b4-chat/snapshots/d8fea83177443cd5a27e55bc52f4b434fd5a7709/ --max_tokens 140000 --min_count 100 --output_dir /dev/shm/violet-1b4/  --vocab ./experiments/runv3mini-eval1-500M/wordgrams.txt
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.7.1+cu126).
[1/4] Загружаю базовый токенизаторъ: /huggingface/hub/models--zakarth--violet-1b4-chat/snapshots/d8fea83177443cd5a27e55bc52f4b434fd5a7709/
       Базовый словарь: 24014
[2/4] Читаю ./experiments/runv3mini-eval1-500M/wordgrams.txt
       Токеновъ къ добавленію: 40274
       Новыхъ (нѣтъ въ словарѣ): 40274
       Уже присутствуютъ: 0
[3/4] Добавляю 40274 токеновъ...
       Фактически добавлено: 40274
       Новый размѣръ словаря: 64288
       Примѣры:
         'и‿нѣ' → [24014]
         'и‿в' → [24015]
         'потому‿что' → [24016]
         'то‿что' → [24017]
         'нѣ‿только' → [24018]
[4/4] Сохраняю: /dev/shm/violet-1b4/
       Готово.

real    0m6,835s
user    0m6,203s
sys     0m0,781s


wordgram_vocab$ time  python ./experiments/wordgram_run1/fix_model_init.py --base_model /huggingface/hub/models--zakarth--violet-1b4-chat/snapshots/d8fea83177443cd5a27e55bc52f4b434fd5a7709/ --max_tokens 140000 --min_count 100 --out_path /dev/shm/violet-1b4/ --wordgram_file ./experiments/runv3mini-eval1-500M/wordgrams.txt --use_gpu
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.7.1+cu126).
[1/4] Загружаю базовую модель: /huggingface/hub/models--zakarth--violet-1b4-chat/snapshots/d8fea83177443cd5a27e55bc52f4b434fd5a7709/
       Режимъ: GPU (torch.float16 для экономіи VRAM)
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████| 316/316 [00:00<00:00, 413.97it/s]
       Старый словарь: 24014
[2/4] Читаю ./experiments/runv3mini-eval1-500M/wordgrams.txt
       Кандидатовъ: 40274
       Новыхъ (нѣтъ въ словарѣ): 40274
       Фактически добавлено: 40274
       Новый словарь: 64288
[3/4] Расширяю эмбеддинги (методъ mean)...
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
[transformers] The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
[4/4] Сохраняю въ /dev/shm/violet-1b4/
Writing model shards: 100%|██████| 2/2 [00:04<00:00,  2.40s/it]
       Тестъ: 'и‿нѣ' → ID [24014]
Готово!

real    0m40,825s
user    0m25,083s
sys     0m4,049s
```
