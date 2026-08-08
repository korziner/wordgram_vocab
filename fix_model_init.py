import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Расширеніе эмбеддинговъ модели")
parser.add_argument("--base_model", required=True, help="Путь къ базовой модели")
parser.add_argument("--wordgram_file", required=True, help="Путь къ wordgrams.txt")
parser.add_argument("--out_path", required=True, help="Куда сохранить результатъ")
parser.add_argument("--max_tokens", type=int, default=100000, help="Макс. число добавляемыхъ токеновъ")
parser.add_argument("--min_count", type=int, default=0, help="Минимальная частота")
parser.add_argument("--use_gpu", action="store_true", help="Принудительно использовать GPU")
args = parser.parse_args()

print(f"[1/4] Загружаю базовую модель: {args.base_model}")

if args.use_gpu:
    print("       Режимъ: GPU (torch.float16 для экономіи VRAM)")
    # Загружаемъ въ float16 и сразу переносимъ на cuda:0
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16).to("cuda:0")
else:
    print("       Режимъ: CPU (безопаснѣе при ресайзѣ, избѣгаетъ OOM)")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)

tok = AutoTokenizer.from_pretrained(args.base_model)
old_vocab = len(tok)
print(f"       Старый словарь: {old_vocab}")

print(f"[2/4] Читаю {args.wordgram_file}")
new_words = []
with open(args.wordgram_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2: continue
        word, freq_str = parts[0], parts[1]
        try:
            freq = int(freq_str)
        except ValueError:
            continue
        if args.min_count > 0 and freq < args.min_count:
            continue
        new_words.append(word)

new_words = new_words[:args.max_tokens]
print(f"       Кандидатовъ: {len(new_words)}")

existing = set(tok.get_vocab().keys())
to_add = [w for w in new_words if w not in existing]
print(f"       Новыхъ (нѣтъ въ словарѣ): {len(to_add)}")

if not to_add:
    print("       Нечего добавлять!")
else:
    added = tok.add_tokens(to_add, special_tokens=False)
    print(f"       Фактически добавлено: {added}")

new_vocab = len(tok)
print(f"       Новый словарь: {new_vocab}")

print("[3/4] Расширяю эмбеддинги (методъ mean)...")
# Эта операція безопасна и на CPU, и на GPU
model.resize_token_embeddings(new_vocab, pad_to_multiple_of=256)

print(f"[4/4] Сохраняю въ {args.out_path}")
os.makedirs(args.out_path, exist_ok=True)

# Сохраняемъ всегда въ float16 или float32, safetensors надёженъ
model.save_pretrained(args.out_path, safe_serialization=True, max_shard_size="2GB")
tok.save_pretrained(args.out_path)

test = "и‿нѣ"
ids = tok.encode(test, add_special_tokens=False)
print(f"       Тестъ: {test!r} → ID {ids}")
print("Готово!")
