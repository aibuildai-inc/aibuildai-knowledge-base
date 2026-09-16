# ByT5-Large with word-level DAPT

Competition: deep-past-initiative-machine-translation
Rank: #30
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/byt5-large-with-word-level-dapt

## 1. Model Choice

- **Architecture:** Google ByT5-Large - byte-level sequence-to-sequence Transformer as many others have chosen, aviod the messy tokenization with Akkadian
- **Base vs Large:** ByT5-base and ByT5-large performed similarly on holdout and public LB (same score for base & large), but large edged out on private
- **Precision:** Always trained in FP32 (BF16 training caused degradation). Inference also in FP32 on the final submission, but FP16 inference probably could have worked


## 2. Data Sources & Preprocessing

**Data sources (~510K clean parallel pairs after filtering):**

All based on materials provided by the competition host. PDFs were first processed with DeepSeek-OCR 2, then used LLM for separating meaningful corpus pairs. Assigned quality tiers of data for later selective training exposure, based on length, number of <gap>, unique word ratio etc.

| Source | Entries | Description |
|--------|---------|-------------|
| Official competition data | 1,561 | Provided by the host as starting point |
| Chicago Assyrian Dictionary (CAD) | 280K | All word level data, but pretty messy |
| Publications (415 papers) | 76K | Majority source of sentence level data |
| Old Assyrian Lexicon (eBL) | 165K | Word-level transliteration/definition pairs |
| AKT batch | 16K | Commercial/legal letter translations |
| Hecker ICK4 letters | 1.4K | Complete letter document translations |
| Kouwenberg grammars | 10K | Grammar examples from 2 textbooks |
| Other (eSAD, onomasticon list from host,etc.) | ~17K | Supplementary dictionaries and glossaries |


**Preprocessing pipeline (9 stages):**
1. **Unicode normalization:** NFC, transliteration variant mapping (ḫ→h, subscripts→digits, diacritics normalized)
2. **Source cleaning:** Determinative types normalized (`{ki}`, `{d}`), editorial markers stripped, English leakage removed, name placeholders (PN/GN/DN) → `<gap>`
3. **Target cleaning:** Shekel fraction conversion, LaTeX removal, citation stripping, grammatical gloss removal, commodity abbreviation expansion, scholarly apparatus removal
4. **Gap unification:** All damage markers (`[...]`, `[x x x]`, `(break)`, half-brackets, ellipses, etc.) → canonical `<gap>` token. Adjacent gaps collapsed. This was a major contributor to score improvement.
5. **Quality filtering:** Removed ~80K entries (empty, high damage ratio, scholarly annotations, self-referential, metalanguage)
6. **Deduplication:** ~13K exact duplicates removed (priority-based: higher-quality source kept)
7. **Long entry splitting:** Entries >512B split via LLM processing at clause boundaries
8. **Back-translation:** Back translated some English translation to Akkadian using a trained ByT5 base model (English→Akkadian), use as synthetic data to broaden the spectrum


## 3. Training Method & Curriculum

**Training curriculum:**

| Stage | Data | LR | Aug | Purpose |
|-------|------|-----|-----|---------|
| DAPT | 114K word-level (unique texts) | 5e-4 | none | Domain adaptation pre-training |
| S1 | 221K word-level glosses | 5e-4 | none | Vocabulary grounding |
| S2 | 282K all sentences | 3e-4 | none | Sentence-level learning |
| S3–S8 | Sentences <512B | 2e-4 → 3e-5 | 0.2 → 0.3 | Progressive fine-tuning with augmentation escalation |

**Synthetic augmentation / selection after S2 with increasing probability:**

- A: Diacritical simplification (š→s, ṣ→s)
- B: Hyphenation merge/split/move
- C: Determinative perturbation
- D: Sign reading number swap (subscript indices)
- E: Spacing variation
- F: Gap augmentation (character/word spans)

**Key training findings:**

- Low–moderate augmentation (0.2–0.4) optimal; high (0.5+) actively hurts
- Hard Example Mining (HEM) not helping: aggressive → reduced score; gentle → nearly neutral


## 4. Inference Settings & Selection Strategy

**Best submission:**

- `num_beams=12, length_penalty=1.1, repetition_penalty=1.2`
- `max_new_tokens=512, do_sample=False`
- **Simple beam top-1 selection** — no MBR, no reranking
- FP32 inference on dual T4 GPUs (workload split round-robin)
- Dynamic batch sizing (base=6 per GPU, scaled inversely by source byte length)
- OOM fallback to batch_size=1
- Source chunking: Split > 512-byte chunks at word boundaries and reassemble with de-dup

**Other approaches explored but did not improve LB:**

*MBR reranking:*

- chrF++ MBR: Better vs. beam top-1 (36% of oracle gap) for my own eval set, but no impact on LB
- Greedy MBR+features: chrF++ (80%) + UWR (20%) → +FWR (10%) → +Period (5%). same, better onmy own eval set but not on LB

*Other things:*

- LogProb reranking
- TTA (test-time augmentation): no impact on my own eval set
- kNN-MT decoding: implemented (adaptive λ, k=16, GPU-based L2 distance) not working
- Per-bucket length penalty: marginal improvement and heavy overhead, dropped


## 5. Postprocessing

Applied after generation:

1. **normalize_gaps** — all gap variants → canonical `<gap>`, adjacent gaps collapsed
2. **bracket removal** — `[text]` → `text`, preserving `<gap>`
3. **gap collapse** — `<gap> <gap>` → `<gap>`
4. **empty fallback** — empty translations → `<gap>.`

Additional postprocessing developed during the project (used in MBR variants):

- fix_capitalization (ByT5 byte-confusion mid-word case)
- fix_common_artifacts, fix_typos (data-driven corrections)
- clean_repetitions + remove_repeated_ngrams (beam search phrase loops)
- remove_scribal_notations, transliterate_akkadian_chars
- convert_decimals_to_fractions, normalize_subscript_digits


## 6. Other Explorations

**Cross-model ensemble:**

- Tried Large + Base ensemble with merged candidate pools → MBR
- Also tried multiple Base ensembles (different ByT5-base checkpoints, curriculms,. data exposure)
- Result: very modest gain on my local eval and all slightly worse in LB, did not end up using
