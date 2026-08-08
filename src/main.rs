// src/main.rs
// ============================================================================
// wordgram_vocab — Извлеченіе частотнаго словаря СЛОВЕСНЫХЪ n-граммъ
//                  (пары, тройки, четвёрки словъ) изъ JSONL-корпуса.
//
// Rust + rayon + bloom filter.
// ============================================================================

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use fxhash::FxBuildHasher;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Аргументы
// ─────────────────────────────────────────────────────────────────────────────

/// Извлеченіе частотнаго словаря словесныхъ n-граммъ изъ JSONL-корпуса.
///
/// Примѣры:
///
///   wordgram_vocab -i corpus.jsonl.zst -o vocab.txt --word-ngrams 2,3
///
///   wordgram_vocab -i corpus.jsonl -o vocab.txt --word-ngrams 2,3,4 --min-count 3
///
///   wordgram_vocab -i corpus.jsonl.zst -o vocab.txt --word-ngrams 2,3 \
///       --separator "▁" --lowercase --verbose
#[derive(Parser)]
#[command(name = "wordgram_vocab", version, about, long_about = None)]
struct Args {
    /// Путь къ JSONL-файлу (обычный или .zst).
    #[arg(short, long)]
    input: PathBuf,

    /// Путь къ выходному TSV (словосочетаніе\tчастота).
    #[arg(short, long)]
    output: PathBuf,

    /// Размѣры словесныхъ n-граммъ чрезъ запятую (2=пары, 3=тройки, 4=четвёрки).
    #[arg(long, default_value = "2,3")]
    word_ngrams: String,

    /// Раздѣлитель словъ въ выходномъ токене.
    #[arg(long, default_value = "_")]
    separator: String,

    /// Минимальная частота для включенія въ словарь.
    #[arg(long, default_value_t = 2)]
    min_count: u64,

    /// Максимальный размѣръ словаря (0 = безъ ограниченіи).
    #[arg(long, default_value_t = 0)]
    max_vocab: usize,

    /// Минимальная длина слова (0 = не фильтровать).
    #[arg(long, default_value_t = 0)]
    min_word_len: usize,

    /// Приводить къ нижнему регистру.
    #[arg(long)]
    lowercase: bool,

    /// Ожидаемое число уникальныхъ n-граммъ (для bloom filter).
    #[arg(long, default_value_t = 20_000_000)]
    expected_unique: usize,

    /// Вѣроятность ложнаго срабатыванія bloom filter.
    #[arg(long, default_value_t = 0.01)]
    fp_rate: f64,

    /// JSON-поле съ текстомъ.
    #[arg(long, default_value = "text")]
    field: String,

    /// Число потоковъ rayon (0 = авто).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Размеръ пачки строкъ.
    #[arg(long, default_value_t = 10_000)]
    chunk_size: usize,

    /// Подробный выводъ.
    #[arg(short, long)]
    verbose: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Bloom filter
// ─────────────────────────────────────────────────────────────────────────────

struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
}

impl BloomFilter {
    fn new(expected_items: usize, fp_rate: f64) -> Self {
        let ln2 = std::f64::consts::LN_2;
        let n = expected_items.max(1) as f64;
        let num_bits = (-(n) * fp_rate.ln() / (ln2 * ln2)).ceil() as usize;
        let num_bits = num_bits.max(64);
        let num_hashes = ((num_bits as f64 / n) * ln2).ceil() as u32;
        let num_hashes = num_hashes.clamp(1, 16);
        let words = num_bits.div_ceil(64);
        BloomFilter { bits: vec![0u64; words], num_bits, num_hashes }
    }

    // ИСПРАВЛЕНО E0502: хеши вычисляются въ Vec ДО записи въ bits
    #[inline]
    fn insert(&mut self, item: &[u8]) {
        let positions: Vec<usize> = self.hash_positions(item);
        for pos in positions {
            self.bits[pos >> 6] |= 1u64 << (pos & 63);
        }
    }

    #[inline]
    fn contains(&self, item: &[u8]) -> bool {
        self.hash_positions(item).iter().all(|&pos| {
            self.bits[pos >> 6] & (1u64 << (pos & 63)) != 0
        })
    }

    // Общій вычислитель позицій (нѣтъ заимствованія self.bits)
    #[inline]
    fn hash_positions(&self, item: &[u8]) -> Vec<usize> {
        let h1 = fxhash::hash64(item);
        let h2 = {
            let mut buf = Vec::with_capacity(item.len() + 1);
            buf.extend_from_slice(item);
            buf.push(0xFF);
            fxhash::hash64(&buf)
        };
        let nb = self.num_bits as u64;
        (0..self.num_hashes)
            .map(|i| {
                (h1.wrapping_add((i as u64).wrapping_mul(h2)) % nb) as usize
            })
            .collect()
    }

    fn union_with(&mut self, other: &BloomFilter) {
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a |= *b;
        }
    }

    fn count_ones(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Кириллица (дореформенныя буквы уже въ діапазонѣ 0400..052F)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn is_cyrillic(c: char) -> bool {
    matches!(c,
        '\u{0400}'..='\u{052F}'   // Кириллица + дополненія (включая Ѣ І Ѳ Ѵ Ѧ Ѫ)
        | '\u{A640}'..='\u{A69F}' // Кириллица расширенная-B
        | '\u{2DE0}'..='\u{2DFF}' // Кириллица расширенная-A
        | '\u{1C80}'..='\u{1C8F}' // Кириллица расширенная-C
    )
}

#[inline]
fn is_word_char(c: char) -> bool {
    is_cyrillic(c) || c == '\'' || c == '-' || c == '\u{2019}'
}

// ─────────────────────────────────────────────────────────────────────────────
// Извлеченіе словъ и словесныхъ n-граммъ
// ─────────────────────────────────────────────────────────────────────────────

fn extract_words(text: &str, lowercase: bool, min_word_len: usize) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if is_word_char(c) {
            if lowercase {
                for lc in c.to_lowercase() {
                    current.push(lc);
                }
            } else {
                current.push(c);
            }
        } else {
            if !current.is_empty() {
                if min_word_len == 0 || current.chars().count() >= min_word_len {
                    words.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }
    if !current.is_empty() {
        if min_word_len == 0 || current.chars().count() >= min_word_len {
            words.push(current);
        }
    }

    words
}

fn word_ngrams(words: &[String], ns: &[usize], sep: &str) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let wlen = words.len();

    for &n in ns {
        if wlen < n { continue; }
        for start in 0..=(wlen - n) {
            let joined = words[start..start + n].join(sep);
            result.push(joined.into_bytes());
        }
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Быстрое извлеченіе JSON-поля
// ИСПРАВЛЕНО E0106: добавлено время жизни 'a
// ─────────────────────────────────────────────────────────────────────────────

fn extract_text_fast<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let pattern = format!("\"{}\"", field);
    let idx = line.find(&pattern)?;
    let rest = &line[idx + pattern.len()..];
    let rest = rest.trim_start().strip_prefix(':')?;
    let rest = rest.trim_start().strip_prefix('"')?;

    let bytes = rest.as_bytes();
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escaped { escaped = false; continue; }
        match b {
            b'\\' => escaped = true,
            b'"' => return Some(&rest[..i]),
            _ => {}
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Локальное состояніе потока
// ─────────────────────────────────────────────────────────────────────────────

type FxMap = std::collections::HashMap<Vec<u8>, u64, FxBuildHasher>;

struct ThreadLocal {
    bloom: BloomFilter,
    map: FxMap,
}

impl ThreadLocal {
    fn new(expected: usize, fp_rate: f64) -> Self {
        ThreadLocal {
            bloom: BloomFilter::new(expected, fp_rate),
            map: FxMap::default(),
        }
    }

    #[inline]
    fn observe(&mut self, key: &[u8]) {
        if self.bloom.contains(key) {
            *self.map.entry(key.to_vec()).or_insert(1) += 1;
        } else {
            self.bloom.insert(key);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let args = Args::parse();

    let ns: Vec<usize> = args.word_ngrams
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if ns.is_empty() {
        eprintln!("ОШИБКА: --word-ngrams не содержитъ чиселъ");
        std::process::exit(1);
    }

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("rayon");
    }

    if args.verbose {
        eprintln!("═══ wordgram_vocab ═══");
        eprintln!("Входъ:            {}", args.input.display());
        eprintln!("Выходъ:           {}", args.output.display());
        eprintln!("Словесныя n-гр.:  {:?}", ns);
        eprintln!("Раздѣлитель:      {:?}", args.separator);
        eprintln!("Мин. частота:     {}", args.min_count);
        eprintln!("Мин. длина слова: {}", args.min_word_len);
        eprintln!("Нижній регистръ:  {}", args.lowercase);
        eprintln!("Bloom:            {} эл., fp={}", args.expected_unique, args.fp_rate);
        eprintln!("Потоковъ:         {}", rayon::current_num_threads());
        eprintln!("═══════════════════════");
    }

    // ── 1. Чтеніе ──
    if args.verbose { eprintln!("[1/4] Читаю..."); }

    let lines: Vec<String> = {
        let file = File::open(&args.input)?;
        let reader: Box<dyn BufRead> =
            if args.input.extension().map_or(false, |e| e == "zst") {
                Box::new(BufReader::with_capacity(1 << 20, zstd::Decoder::new(file)?))
            } else {
                Box::new(BufReader::with_capacity(1 << 20, file))
            };
        reader.lines().collect::<Result<Vec<_>, _>>()?
    };

    let total_lines = lines.len();
    if args.verbose { eprintln!("       Строкъ: {}", total_lines); }

    // ── 2. Параллельная обработка ──
    if args.verbose {
        eprintln!("[2/4] Извлекаю словесныя n-граммы ({:?})...", ns);
    }

    let field = args.field.clone();
    let sep = args.separator.clone();
    let lowercase = args.lowercase;
    let min_word_len = args.min_word_len;
    let expected = args.expected_unique;
    let fp_rate = args.fp_rate;
    let ns_ref = ns.clone();
    let min_n = *ns_ref.iter().min().unwrap_or(&2);

    // ИСПРАВЛЕНО E0308: reduce возвращаетъ ThreadLocal, не кортежъ
    let merged: ThreadLocal = lines
        .par_chunks(args.chunk_size)
        .fold(
            || ThreadLocal::new(expected / rayon::current_num_threads().max(1), fp_rate),
            |mut local, chunk| {
                for line in chunk {
                    let text = if line.starts_with('{') {
                        extract_text_fast(line, &field).unwrap_or(line.as_str())
                    } else {
                        line.as_str()
                    };
                    if text.is_empty() { continue; }

                    let words = extract_words(text, lowercase, min_word_len);
                    if words.len() < min_n { continue; }

                    let grams = word_ngrams(&words, &ns_ref, &sep);
                    for g in &grams {
                        local.observe(g);
                    }
                }
                local
            },
        )
        .reduce(
            || ThreadLocal::new(1, fp_rate),
            |mut acc, other| {
                acc.bloom.union_with(&other.bloom);
                for (k, v) in other.map {
                    *acc.map.entry(k).or_insert(0) += v;
                }
                acc
            },
        );

    if args.verbose {
        let fill = merged.bloom.count_ones() as f64
            / (merged.bloom.bits.len() as f64 * 64.0);
        eprintln!("       Bloom: {:.1}% заполненъ", fill * 100.0);
        eprintln!("       HashMap: {} уникальныхъ (частота ≥ 2)", merged.map.len());
    }

    // ── 3. Фильтрація и сортировка ──
    if args.verbose { eprintln!("[3/4] Фильтрую (min_count={})...", args.min_count); }

    let mut items: Vec<(Vec<u8>, u64)> = merged.map
        .into_iter()
        .filter(|(_, cnt)| *cnt >= args.min_count)
        .collect();

    items.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    if args.max_vocab > 0 {
        items.truncate(args.max_vocab);
    }

    // ── 4. Запись ──
    if args.verbose { eprintln!("[4/4] Записываю {} токеновъ...", items.len()); }

    let out_file = File::create(&args.output)?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);

    for (token_bytes, count) in &items {
        let token = String::from_utf8_lossy(token_bytes);
        writeln!(writer, "{}\t{}", token, count)?;
    }
    writer.flush()?;

    // ── Статистика ──
    if args.verbose {
        eprintln!("═══════════════════════");
        eprintln!("Строкъ:            {}", total_lines);
        eprintln!("Всего токеновъ:    {}", items.len());

        let sep_bytes = args.separator.as_bytes();
        for &n in &ns {
            let expected_seps = n - 1;
            let count = items.iter().filter(|(tok, _)| {
                tok.windows(sep_bytes.len())
                    .filter(|w| *w == sep_bytes)
                    .count() == expected_seps
            }).count();
            let label = match n {
                2 => "пары",
                3 => "тройки",
                4 => "четвёрки",
                _ => "n-граммы",
            };
            eprintln!("  {} ({} словъ): {}", label, n, count);
        }

        if let Some((tok, cnt)) = items.first() {
            eprintln!("Частѣйшій:  {} ({})", String::from_utf8_lossy(tok), cnt);
        }
        if let Some((tok, cnt)) = items.last() {
            eprintln!("Рѣдчайшій:  {} ({})", String::from_utf8_lossy(tok), cnt);
        }
        eprintln!("Выходъ:     {}", args.output.display());
    }

    Ok(())
}
