# [2nd Place] Data-Centric Akkadian NMT

Competition: deep-past-initiative-machine-translation
Rank: #2
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/2nd-place-data-centric-akkadian-nmt

# [2nd Place] Data-Centric Akkadian NMT: LLM-Assisted Sentence Alignment + Large-Scale Academic PDF Mining

**Score: 41.9 (Public) / 41.0 (Private) | Team: wukeneth | Model: google/byt5-large**

---

## TL;DR

A fully data-centric solution. The model is vanilla **ByT5-large** with no architectural changes. Every point of improvement came from building a larger, cleaner training corpus through:

1. A three-stage LLM pipeline to produce sentence-level alignments from the official data
2. Manually sourcing and OCR-extracting ~60 academic publications on Old Assyrian cuneiform not covered by the official dataset
3. Text normalization that unifies the highly variable orthography across publications

---

## The Problem

The competition provides three key files:

- **train.csv** — ~6,500 documents with full English translations, but **no sentence-level alignment**
- **published_texts.csv** — ~13,456 published texts, many untranslated
- **Sentences_Oare_FirstWord_LinNum.csv (S_meta)** — pre-existing sentence fragments for ~1,700 documents, but the word indices and translations are **frequently misaligned or corrupt**

Training directly on full-document pairs (Akkadian full text → full English translation) works but leaves significant value on the table. The model needs clean **sentence-level pairs** to learn tight correspondences. The core challenge was building those pairs at scale.

---

## Data Pipeline: Three-Stage LLM Segmentation

I partitioned all published texts into three non-overlapping sets and applied a different LLM strategy to each. The three sets are defined by strict set arithmetic on the official data files:

| Set           | Definition                 | LLM Strategy                                            |
| ------------- | -------------------------- | ------------------------------------------------------- |
| **Breaker**   | (U_train ∪ U_pub) − S_meta | Segment full translation into sentence-level pairs      |
| **Fixer**     | S_meta                     | Re-anchor corrupt fragments using OARE API word indices |
| **Generator** | U_pub − (U_train ∪ S_meta) | Synthesize translation from morphological metadata      |

### Set 1 — Sentence Breaker (clean full-text alignment)

**Input**: Akkadian transliteration + full English translation from `train.csv` + OARE API word indices and physical line numbers.

**Method**: For each document, the OARE ID is used to look up per-word dictionary entries (normalized form, lemma) from the OARE word table. An LLM is then prompted with: the full Akkadian text, the per-word dictionary context, and the full English translation. It splits the English translation into sentence-sized chunks and assigns a `word_range` (start/end word index) to each chunk using the physical word coordinates as anchors. Because the source texts are clean (no pre-existing corrupt segmentation), the output is reliable.

**Result**: 1,558 documents → **9,378 sentence pairs**

### Set 2 — Translation Fixer (misalignment correction)

**Input**: Corrupt S_meta fragments + OARE API
ormalized_form, word_index and line_number fields for each word.

**Method**: Use the existing fragments as a _weak reference only_. Re-anchor each fragment by matching normalized word forms from the API to the physical tablet structure. The LLM reconstructs correct sentence boundaries from scratch, using the S_meta entries only as coarse hints.

**Result**: 1,416 documents → **11,302 sentence pairs**

### Set 3 — Translation Generator (attempted, did not help)

Attempted to generate pseudo-translations using only morphological metadata (G-Stem, 3ms, Preterite, dictionary entries). The outputs were too noisy and inconsistent. Including them in training **hurt the score** — dropped entirely.

---

## External Data: Official PDFs + Independent Literature Mining

This was the single biggest score improvement. In addition to the competition-provided PDF corpus (AKT series, large monographs), I independently located, downloaded, OCR-processed, and extracted dozens of additional Old Assyrian cuneiform text editions from academic journals and open-access repositories.

PDF-extracted sources already carry sentence-level structure (each tablet entry in a publication presents transliteration and translation line-by-line or sentence-by-sentence). The LLM extraction pipeline preserves this structure automatically — no separate sentence-breaking step is needed for external data.

### Sources covered

| Category                   | Key sources                                                                      |
| -------------------------- | -------------------------------------------------------------------------------- |
| AKT series                 | AKT 1–4, 7A/B, 9A/B, 10, 11A/B, 12 (AKT 5/6A-6E/8 already in official train.csv) |
| ArAn (Archivum Anatolicum) | ~15 articles (albayrak, cecen, erol, kuzuoglu, etc.)                             |
| Belleten                   | ~10 articles (bayram, donbaz, gunbatti, sever, etc.)                             |
| DTCF Dergisi               | ~8 articles                                                                      |
| ATDD / TAD                 | ~6 articles                                                                      |
| HAL open access            | French institutionally-archived papers                                           |
| International journals     | JCS, JNES, Anatolica, JEOL, AoF, RA, NABU, etc.                                  |
| Large monographs           | Michel LAPO19, Veenhof PIHANS 111, Barjamovic PIHANS 120, OIP27, etc.            |
| Misc                       | Women of Assur and Kanesh, ATHE Kienast, Assur-Nada 2002, etc.                   |

**~60,654 external sentence pairs** from **149 distinct sources** after cleaning.

### Language distribution and non-English handling

| Language     | Sentence pairs |
| ------------ | -------------- |
| Turkish (TR) | 27,089         |
| English (EN) | 21,683         |
| French (FR)  | 6,436          |
| German (DE)  | 5,413          |
Dataset is in the attachment below.

Many Turkish- and German-language papers cover the same Akkadian tablets as English publications. During extraction, the LLM **translated non-English text directly into English**, so all training pairs remain `Akkadian → English`. This adds more Akkadian surface forms to training without introducing a multi-target problem.

When the same physical tablet (identified by IDs like `Kt 88/k 71`, `CCT 4 12a`) appears in both an English and a Turkish/German publication, only the English version is kept (_Prefer-EN dedup_), preventing conflicting translations for the same source.

Turkish-language sources are already amplified up to **3×** through multi-version extraction (main + two backup runs), while fresh English-language academic papers have no backup copies — only a single extraction version. To compensate for this asymmetry, English-language academic papers are explicitly **boosted 2×** so they remain competitive in training exposure alongside the naturally-amplified Turkish sources.

### Multi-version extraction

The LLM-based extraction pipeline was executed **three separate times** on each non-English source (Turkish, German, French, etc.), at different points in time. Each run independently translated the source text (TR/DE/FR → EN) and stored its outputs in a separate database directory (`main`, `Backup_20260227`, `Backup_manual`). Because LLM outputs vary across runs, the three passes yield slightly different English renditions for the same Akkadian input.

The dataset generator includes **all three versions** for any source that has backup rows (`is_backup=1`), providing implicit data augmentation through natural translation variance across extraction runs. Sources with no backup rows (single extraction only) use just the main version.

Sources already covered by the Sentence Breaker / Fixer pipeline (AKT 5, 6A–6E, 8 — English-language and forming the core of `train.csv`) were not given backup extraction copies. Running additional passes on these would produce near-duplicates of the breaker/fixer sentence pairs already in the dataset, adding no new information.

---

## Text Normalization

Akkadian transliteration orthography is highly inconsistent across publications. Without normalization, the same cuneiform sign appears in dozens of surface variants.

Key rules (all applied to source side):

| Issue                             | Rule                             |
| --------------------------------- | -------------------------------- |
| sz / s, + consonant               | → š                              |
| , + consonant                     | → ṭ                              |
| Subscript 2 after vowel (a2, e2...)  | acute accent: a2→á, e2→é, i2→í, u2→ú |
| Subscript 3 after vowel (a3, e3...)  | grave accent: a3→à, e3→è, i3→ì, u3→ù |
| Higher subscripts (₄₅₆...)        | → plain ASCII digits             |
| Hamza variants (ʾ, ʿ, ʼ)          | → deleted                        |
| ḫ / Ḫ                             | → h / H (competition convention) |
| Determinatives (d), (ki), etc.    | → {d}, {ki} (cuneiform standard) |
| Other edge cases                  | additional rules in `text_normalization.py` |
The script of data cleaning is attached in the links below.

---

## Document-Level Augmentation (768-byte chunks)

In addition to sentence-level pairs, consecutive sentences are concatenated into **document-level training examples** up to a 768-byte (UTF-8) ceiling. This:

- Provides inter-sentence context during training
- Creates implicit augmentation via variable chunking boundaries
- Matches max_source_length = 768 of ByT5-large exactly (no truncation)

---

## Training Configuration

### Run 1 — Submitted (Public 41.8 / Private 41.0, ~90,522 pairs)

| Parameter            | Value                                                             |
| -------------------- | ----------------------------------------------------------------- |
| Model                | `google/byt5-large`                                               |
| Max sequence length  | 768 bytes                                                         |
| Learning rate        | 7e-5                                                              |
| Optimizer            | Adafactor (β₁=None, decay_rate=-0.8, clip=1.0, eps=(1e-30, 1e-3)) |
| Effective batch size | 32 × 4 grad accumulation = 128                                    |
| Epochs               | 16                                                                |
| Warmup steps         | 800                                                               |
| LR scheduler         | Linear                                                            |
| Weight decay         | 0.01                                                              |
| Max grad norm        | 10.0                                                              |
| Group by length      | Yes (sorts by input length to minimize padding waste)             |
| Save epochs          | 14, 15, 16 (epoch 15 selected — highest score)                    |

### Run 2 — Same dataset, alternative config (Public 42.1 / Private 40.9, ~90,522 pairs)

| Parameter            | Value                                                            |
| -------------------- | ---------------------------------------------------------------- |
| Model                | `google/byt5-large`                                              |
| Max sequence length  | 768 bytes                                                        |
| Learning rate        | 2e-4                                                             |
| Optimizer            | Adafactor (β₁=0.9, decay_rate=-0.8, clip=1.0, eps=(1e-30, 1e-3)) |
| Effective batch size | 32 × 8 grad accumulation = 256                                   |
| Epochs               | 8 (target ~5,500 steps)                                          |
| Warmup ratio         | 1.1 epoch                                                        |
| LR scheduler         | Linear                                                           |
| Weight decay         | 0.01                                                             |
| Max grad norm        | 1.0                                                              |
| Group by length      | Yes                                                              |

### Run 3 — Best config (Public 42.4 / Private 40.4, ~100,143 pairs)

| Parameter            | Value                                                             |
| -------------------- | ----------------------------------------------------------------- |
| Model                | `google/byt5-large`                                               |
| Max sequence length  | 768 bytes                                                         |
| Learning rate        | 7e-5                                                              |
| Optimizer            | Adafactor (β₁=None, decay_rate=-0.8, clip=1.0, eps=(1e-30, 1e-3)) |
| Effective batch size | 32 × 4 grad accumulation = 128                                    |
| Epochs               | 14 (target ~11,000 steps)                                         |
| Warmup ratio         | 1.1 epoch                                                         |
| LR scheduler         | Linear                                                            |
| Weight decay         | 0.01                                                              |
| Max grad norm        | 10.0                                                              |
| Group by length      | Yes                                                              |

### Note on group_by_length and gradient stability

`group_by_length=True` batches samples of similar input length together, significantly reducing padding waste and speeding up training. However, it introduces a side effect: consecutive batches within the same length bucket have very similar sequence lengths, which creates **non-uniform gradient magnitudes** across steps — batches of long sequences produce much larger raw gradients than batches of short ones, causing instability in the loss curve.

The fix was setting `β₁=0.9` in Adafactor. By default, Adafactor uses no first-moment estimate (β₁=None). Adding β₁=0.9 introduces a momentum term that carries a weighted average of recent gradients, providing inertia that dampens the sharp per-step variance introduced by length-grouped batching. This stabilized training without sacrificing the throughput gains from `group_by_length`.

| Dataset version                  | Training pairs | Public LB | Private LB |
| -------------------------------- | -------------- | --------- | ---------- |
| Run 1 — Submitted                | ~90,522        | 41.8–41.9 | 40.9–41.0  |
| Run 2 — Same dataset, alt config | ~90,522        | 42.1      | 40.9       |
| Run 3 — Best config (more data)  | ~100,143       | 42.4      | 40.4       |

Run 2 used the same dataset as Run 1 but with a different hyperparameter configuration, achieving a higher public score (42.1 vs 41.9). Run 3 added more data (~100k pairs) and reached the highest public score (42.4), though with a larger public/private gap (40.4 private). Run 1 was selected as the final submission based on its more stable public/private alignment.

---

## What Worked

| Technique                                   | Impact                             |
| ------------------------------------------- | ---------------------------------- |
| LLM-based sentence breaker/fixer pipeline   | **High** — core training data      |
| External academic PDF mining                | **High** — biggest data expansion  |
| Text normalization (sz/s,/ṭ, subscripts, ḫ) | **Medium** — removes surface noise |
| 768-byte document augmentation              | **Medium** — contextual signal     |
| English prioritization (prefer-EN dedup)    | **Small positive**                 |
| Multi-language (TR/DE) augmentation         | **Small positive**                 |
| English academic papers 2× boost            | **Small positive**                 |

## What Didn't Work

- **Translation Generator**: LLM translations from morphological metadata alone (no reference English) were too noisy; including them hurt scores.
- **Word-level dictionary augmentation**: Adding `translate Akkadian word to English: awilum → man, person` pairs as training samples showed no benefit.
- **PN/GN post-processing**: Built a fuzzy-matching post-processor (OARE eBL dictionary, ~18k entries) to replace hallucinated name translations with correct transliterated forms. No meaningful improvement — likely because the test set comes from diverse sources with inconsistent name conventions, so no single normalization scheme aligns well; the matching logic was also complex enough to introduce its own errors. In hindsight, if the source OCR is correct, the model already learns the proper name forms from training data and post-hoc correction is unnecessary.

---

## Summary

This competition is a **data bottleneck problem**. The Akkadian→English dataset is tiny by NMT standards. My solution treats data construction as the primary engineering challenge: a systematic LLM pipeline for sentence alignment across ~3,000 documents, combined with large-scale external data mining from the academic literature on Old Assyrian cuneiform (~60 publications). The model itself is standard ByT5-large with no structural modifications.

The key unrealized opportunity: the public/private LB gap indicated I should have kept collecting more external data rather than tuning the model. The private leaderboard reward clean, diverse data much more than architectural tricks.
