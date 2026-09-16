# [9th] Place Solution for the Deep Past Challenge Competition

Competition: deep-past-initiative-machine-translation
Rank: #9
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/9th-place-solution-for-the-deep-past-challenge-c

Thank you to Kaggle and the Deep Past Initiative team for hosting this competition. Translating 4,000-year-old merchant letters from Old Assyrian cuneiform was a unique challenge — part NLP, part archaeology, part data engineering. Special thanks to @deeppast for the active engagement throughout, the dataset updates, and the supplementary resources that made this possible.

## Context

- **Code:** https://github.com/Eleftheria14/Deep_Past_Gold_Solution
- **Business context:** https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview
- **Data context:** https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/data

## TL;DR

9th out of 2,673 teams (solo). 1st place: 42.90 private. This solution: **40.1 private**. Gold medal.

- **Data extraction and cleaning was the biggest differentiator**
- Extracted ~95K translation pairs from publications.csv (~950 PDFs) using Gemini batch API
- 30+ iterative OCR cleaning passes with automated invariants as quality gates
- ByT5-base in FP32 — byte-level precision matters for diacritics like š ṣ ṭ ḫ
- Bidirectional training (each pair trained Akkadian→English and English→Akkadian)
- 4-model ensemble with source-grounded MBR (Minimum Bayes Risk) scoring: **39.2 public → 40.1 private**

## Pipeline Overview



## Data: Extraction, Cleaning, and Sources

The competition provided `publications.csv` containing ~950 OCR'd scholarly PDFs — the single biggest untapped resource. Most contain Akkadian transliterations alongside English (and German/French) translations that aren't in `train.csv`. The core challenge was getting this data out cleanly.

### Extracting Pairs with Gemini

The host pre-labelled each page with a `has_akkadian` flag. This was used to filter pages initially, though the flag wasn't always reliable — additional scanning was needed to find Akkadian content on unlabelled pages.

Pages were chunked by document density — ~5K characters for dense tablet editions, ~12K for mixed-content studies — overlapping by 2 pages to avoid splitting pairs at boundaries. Each chunk was sent to Gemini 2.0 Flash Lite via the Google AI Batch API (with Gemini 2.0 Flash on Vertex AI for Vision OCR and sentence splitting) with a structured extraction prompt:

```
You are an expert Assyriologist extracting Akkadian translation pairs.

## TASK
Extract SENTENCE-LEVEL Akkadian transliteration + translation pairs.

**CRITICAL: Extract FULL SENTENCES, not individual words or names!**

## TRANSLATION FIELDS

For each extraction:
- `translation_original`: The translation EXACTLY as written in the source
  (German, French, English, etc.)
- `translation_language`: Detected language code (en, de, fr, it, other)
- `translation`: English version (copy if already English, translate if
  German/French/etc.)

Example (German source):
  {"transliteration": "a-na {m}Šalim-Aššur qí-bi-ma",
   "translation_original": "Sprich zu Šalim-Aššur",
   "translation_language": "de",
   "translation": "Speak to Šalim-Aššur"}

**If NO sentence-level pairs exist, return empty array:**
  {"extractions": []}

## DETERMINATIVES

Use: {d} (divine), {ki} (place), {m} (male), {f} (female)

**Theophoric names (use {d}):** Aššur, Nabû, Šamaš, Ištar, Marduk,
  Sin, Nergal
**NOT names:** DUMU, KÙ.BABBAR, É, MA.NA, GÍN (Sumerograms)

## RULES
1. Extract SENTENCES only - minimum 3+ words
2. Skip name-only extractions
3. Preserve original translation exactly, then provide English
4. is_complete: false if sentence is cut off

## OUTPUT: JSON matching this schema:
{EXTRACTION_SCHEMA}

## TEXT:
{text}
```

Over half of the extracted pairs had non-English originals (29% German, 17% French, 8% Turkish, and others) — the prompt asked Gemini to preserve the original and provide an English version. This produced **~95K sentence-level pairs** — a ~60× increase over the 1,561 pairs in `train.csv`.



The extracted data matched the competition distribution well on surface metrics: similar length profiles, high feature overlap, and 77% of competition vocabulary covered. However, ~42% of extracted pairs were from non-merchant domains (omens, rituals, dictionary entries, literary texts) while the competition tests Old Assyrian merchant correspondence exclusively.

### Gemini Vision for Scanned PDFs

Some host-provided PDFs had no text layer (scanned images only) — the batch text extraction pipeline couldn't touch these. For these, Gemini Vision was used to OCR the pages directly, then the same extraction and cleaning pipeline ran on the output.

10 OCR methods were evaluated for Akkadian scholarly text — EasyOCR, RapidOCR (PaddleOCR), IBM Docling, pdftotext, and CuReD (an Akkadian-trained Kraken model), among others. All generic OCR engines produced zero Akkadian diacritics (no `š`, `ḫ`, `ṣ`, `ṭ`). CuReD recovered some diacritics but hallucinated `ḫ` into English words (~92% false positive rate — `the` → `tḫe`, `with` → `witḫ`). Gemini Vision was the only method with high diacritic accuracy and zero false positives. Docling was used for layout extraction on scanned sources like Prague ICK 4, then Gemini extraction ran on top of its output.

For PDFs that already had a host text layer, re-running Vision was also tried. The extraction quality was better, but after extensive cleaning of the host OCR it didn't translate to improved LB scores.

### Cleaning & Normalisation

Raw extracted pairs needed both OCR repair and format normalisation to match the competition's expected format. **Claude Code** (with Claude Opus 4.6) was used for writing the fix scripts and for quality judging.

The workflow was an iterative loop: sample a random batch of pairs, send them to Claude Opus for quality judging (scoring each pair on alignment accuracy, OCR quality, and format compliance), identify the systematic error category, write a programmatic fix, add an invariant test, run across the full dataset, check for regressions — and repeat until Claude consistently scored the data as clean.

30+ passes through this loop produced fixes like:

| Before | After | Fix type |
|--------|-------|----------|
| **Diacritical OCR recovery** | | |
| `sa₂-ri-im` | `ša₂-ri-im` | `s`→`š` |
| `tup-pu-šu` | `ṭup-pu-šu` | `t`→`ṭ` |
| `is-tù` | `iš-tù` | `s`→`š` |
| `ha-bu-lam` | `ḫa-bu-lam` | `h`→`ḫ` |
| `fa₂-ri-im` | `ṣa₂-ri-im` | `f`→`ṣ` |
| `a.na` | `a-na` | dot→hyphen |
| **Sumerogram repair** (Sumerian logograms embedded in Akkadian, e.g. KÙ.BABBAR = "silver") | | |
| `kù.babbar` | `KÙ.BABBAR` | Case normalisation |
| `kb` | `KÙ.BABBAR` | Abbreviation expansion |
| `u t u` | `UTU` | Spaced-out OCR recovery |
| `GIN` | `GÍN` | Missing diacritic |
| **Determinative recovery** (semantic classifiers prefixed to names, e.g. {d} = divine) | | |
| `lúSIPA` | `{lú}SIPA` | Prefix → curly bracket |
| `itiGU₄` | `{iti}GU₄` | Prefix → curly bracket |
| `(d)Aššur` | `{d}aššur` | Paren → curly, lowercase after det |
| `(URU)` | `{uru}` | Uppercase expanded → lowercase curly |
| **Lexicon-based repair** (using OA Lexicon + curated corpus lexicon) | | |
| `ašam` | `a-ša-am` | Dehyphenation via lexicon lookup |
| `o`-substituted tokens | correct form | Norvig spell-checker against curated lexicon |
| `1GI` | `IGI` | OCR `1`→`I` in witness formulas |
| **Name and target repair** | | |
| `Íalim-Aššur` | `Šalim-Aššur` | Name OCR (`Í`→`Š`) |
| `Assur` | `Aššur` | Target name normalisation |
| `sheqels` | `shekels` | English typo fix |
| `the the silver` | `the silver` | Repeated word removal |
| **Format normalisation** | | |
| `a-na a2-šur` | `a-na a₂-šur` | ASCII subscripts → Unicode |
| `x x x` | `<gap>` | All gap types unified |
| `0.5 mina` | `½ mina` | Decimals → Unicode fractions |
| `˹a-na˺ [x] bi-tim` | `a-na <gap> bi-tim` | Strip half-brackets, `[x]` → `<gap>` |
| `a -na` | `a-na` | Space-hyphen removal |
| `a--na` | `a-na` | Double-hyphen cleanup |
| `( word )` | `(word)` | ORACC spaced parentheses |
| **Removal** (last resort) | | |
| German/French targets | removed | Language detection |
| Transliteration echoed as target | removed | Character similarity + low English word ratio |

&nbsp;

The diacritical recovery deserves explanation. Akkadian has phonemes like `š`, `ḫ`, `ṭ`, `ṣ` that are distinct from `s`, `h`, `t`, `s` — they produce different words. OCR engines trained on Latin script systematically drop these diacritics. The fix uses corpus frequency ratios: count how often each form appears across the full dataset. If `ša₂` appears 200 times but `sa₂` only 3 times, the rare form is almost certainly OCR damage — the correct spelling is overwhelmingly `ša₂`. The high ratio (≥20×) makes this safe; ambiguous cases (ratio <5×) were left untouched.

Key philosophy: **repair first, remove only as last resort**. Aggressive removal had high false positive rates. Careful repair preserved training data.

Every cleaning rule was encoded as a testable invariant — no plain `h` in source (must be `ḫ`), curly bracket determinatives only, Unicode fractions in targets, English-only targets, no impossible consonant clusters. These ran automatically across all datasets after every cleaning pass, catching regressions before they reached training.

### Tracing One Letter Through the Full Pipeline

Here's one letter (Larsen 2002, p.129, Letter 57 — KTS 1, 13b) traced through the pipeline. This is a 4,000-year-old debt collection letter from an Assyrian merchant demanding repayment. The rows highlighted below are illustrative examples — the full text of every letter went through the pipeline.



The same letter through each pipeline stage:

| PDF (ground truth) | Host OCR (raw) | Host → cleaned | Vision OCR (raw) | Vision → extracted pair |
|---|---|---|---|---|
| `um-ma A-šùr-na-da-/m[a]` | `um-ma A-sùr-na-da-/m[a]` | `um-ma A-šùr-na-da-ma` | `um-ma A-šùr-na-da-/m[a]` | `um-ma A-šùr-na-da-ma` |
| `ù Ší-ša-ah-<šu->ša-ar` | `ù Sí-sa-ah-<su->sa-ar` | `ù Ší-ša-aḫ-šu-ša-ar` | `ù Ší-ša-ah-<šu->ša-ar` | `ù Ší-ša-aḫ-šu-ša-ar` |
| `qí-bi₄-ma 17 GÍN` | `qí-bi7-ma 39"GIN` | `qí-bi₇-ma 39GÍN` | `qí-bi₄-ma 17 GÍN` | `qí-bi₄-ma 17 GÍN` |
| `KB ṣa-ru-pá-am` | `MD"sa-ru pá-am` | `MDṣa-ru pá-am` | `KB ṣa-ru-pá-am` | `KB ṣa-ru-pá-am` |
| `ší-im a-bar-ni-im` | `sí im a-bar-ni-im` | `ší im a-bar-ni-im` | `ší-im a-bar-ni-im` | `ší-im a-bar-ni-im` |
| `iš-tù 10 ša-na-tim` | `is-tù 32"sa-na-tim` | `iš-tù 32ša-na-tim` | `iš-tù 10 ša-na-tim` | `iš-tù 10 ša-na-tim` |
| `ha-bu-lam` | `ha-bu-lam` | `ḫa-bu-lam` | `ha-bu-lam` | `ḫa-bu-lam` |

**Translation:** *"From Aššur-nādā to Iddin-Suen and Šišahšušar: Kuzuziya has owed me 17 shekels of refined silver, the price of an abarniu-textile, for 10 years. How come that when the maid went, he insulted her? Make him pay that silver, and give it to the maid."*

### Sentence Splitting and Additional Sources

The competition trains on document-level pairs but tests on sentence-level. Gemini was used to split document-level translations into sentence-aligned pairs, roughly tripling the competition training data (1,561 → ~5.4K pairs). The English side was first split into sentences using punctuation, then Gemini identified where each sentence's Akkadian source begins.

For example, this document-level pair — a copper trade dispute — is a single training example with 6 sentences mixed together:

> **src:** `a-na PUZUR₄-{d}IM lá-dí-a en-um-a-šur en-na-sú-in a-šur-i-dí i-ku-pí-a ù li-lá qí-bi-ma um-ma en-um-a-šùr ù a-lá-ḫu-um-ma i-na 40 GÚ URUDU SIG₅ ša i-dí-a-šur um-me-a-ni e-pu-lu ša 22 GÚ URUDU SIG₅ ša li-bi₄ a-lá-ḫi-im KÙ.BABBAR-áp-šu a-na-kam né-nu um-me-a-ni nu-ṭá-ib ŠÀ.BA ša 22 GÚ URUDU qá-ta-tí-šu-nu ì-lí-a i-tur₄-DINGIR ù i-dí-ku-bu-um KÙ.BABBAR-áp-šu il₅-qé-ú ší-tí URUDU 18 GÚ 6 ma-na URUDU i li-bi i-dí-a-šur a-lá-an a-nim mì-ma i li-bi-šu ú-lá i-ba-ší šu-ma a-na qá-ta-tí-šu-nu en-um-a-šur ù i-ku-pí-a i-ma-ṣí-ú qá-ta-tí-šu-nu li-il₅-qé-ú-ma ší-bi šu-uk-na-šu-nu-tí bé-lu-ni a-tù-nu ú tí-ir-ta-ku-nu li-li-kam ù bé-lu KÙ.BABBAR ma-du-tum ša lá il₅-qé-ú-ni a-ḫu-ru`
>
> **tgt:** *"To Puzur-Adad, Ladiya, Ennam-Aššur, Enna-Suen, Aššur-idī, Ikūn-pīya and Lila from Ennam-Aššur and Ali-ahum: Of the 40 talents of good copper about which Iddin-Aššur reached an agreement with the investors, we have here satisfied in silver the investors for the 22 talents of good copper that was in Ali-ahum's possession. Thereof, for the 22 talents of copper, their shares, Iliya, Itūr-ilī and Iddin-Kūbum have received its value in silver. The rest of the copper, 18 talents 6 minas of copper, is in Iddin-Aššur's possession. Apart from this nothing is in his possession. If Ennam-Aššur and Ikūn-pīya take responsibility for their shares, they may receive their shares, but set witnesses for them. Our dear lords, also, send us word, for a great many creditors who have received nothing remain."*

Splitting produces 6 focused sentence-level pairs. One example:

> **src:** `ší-tí URUDU 18 GÚ 6 ma-na URUDU i li-bi i-dí-a-šur` → **tgt:** *"The rest of the copper, 18 talents 6 minas of copper, is in Iddin-Aššur's possession."*

The splitting prompt:

```
You are helping split an Akkadian transliteration into
sentence-aligned segments.

I have already split the English translation into {n_sents} sentences:
{english_sentences}

The full Akkadian source is:
{src}

For each English sentence, tell me the FIRST WORD (or first 2-3
hyphenated syllables) of the corresponding Akkadian source segment.
The segments must be contiguous and cover the entire source.

RULES:
- Sentence 1 always starts at the very beginning of the source.
- Each subsequent sentence starts where the previous one ended.
- Akkadian has NO punctuation. Use meaning to find where each English
  sentence's source begins.
- Common patterns: "um-ma X-ma" starts speech, "KIŠIB" starts seal
  formulas, "IGI" starts witness lists.
- If you cannot confidently split, respond with just:
  {"uncertain": true}

Respond with a JSON object:
  {"anchors": ["first_word_1", "first_word_2", ..., "first_word_N"]}
The first anchor MUST match the beginning of the source text.
Each anchor should be 1-3 tokens (hyphenated syllables count as
one token).
```

Beyond publications.csv, several other sources contributed to the final training data:

| Source | Pairs | Period | Extraction method |
|--------|------:|--------|-------------------|
| PDF extraction (publications.csv) | 95,925 | Mixed (mostly OA) | Gemini batch on host OCR + Vision for scanned PDFs |
| AKT volumes (13 tablet editions) | 15,602 | Old Assyrian | PyMuPDF text extraction + Gemini batch |
| Akkademia RINAP | 33,239 | Neo-Assyrian | Pre-aligned line-level corpus |
| OARE sentence pairs | 7,011 | Old Assyrian | Fuzzy word-boundary alignment against published_texts.csv |
| Competition train (sentence-split) | ~5,400 | Old Assyrian | Gemini sentence splitting (from 1,561 doc-level) |
| Competition train (doc-level) | 1,561 | Old Assyrian | As provided |
| Larsen 2002 | 1,416 | Old Assyrian | Gemini extraction from merchant archive PDF |
| secondary_sources.csv | 1,418 | Mixed | Gemini extraction from 2.6M lines OCR |
| ORACC parallel corpus | 1,039 | Mixed | Pre-aligned parallel corpus |
| Prague ICK 4 | 885 | Old Assyrian | Docling OCR + Gemini extraction |
| eSAD dictionary | 143 | Old Assyrian | Dictionary example extraction |
| **Total real pairs** | **~163K** | | |

### Synthetic and Augmented Data

Beyond real scholar-translated pairs, two types of augmented data were generated — new training pairs created by systematically modifying existing ones:

**Gap augmentation** (~124K pairs) — The competition test data contains `<gap>` markers where cuneiform is damaged or missing. To teach models to handle these, Gemini was used to create aligned gaps: given a clean pair, it identifies which source tokens correspond to which target words, deletes aligned spans from both sides, and replaces them with `<gap>`. A glossary of known Sumerogram→English mappings, name translations, and dictionary entries was provided to ground the alignment. Used in Models B and C.

```
We are training an Akkadian-English translation model to handle damaged
cuneiform tablets. When parts of a tablet are broken, both the Akkadian
source and English translation have missing sections marked with <gap>.

Your job: take a COMPLETE pair and simulate damage by DELETING aligned
spans from BOTH sides, replacing them with <gap>.

Work in three steps:

STEP 1 — ALIGN: Figure out which source tokens correspond to which
         target words. List the alignment.

STEP 2 — MASK: Pick source tokens to delete. Find the EXACT English
         words from the target that correspond to them (using your
         alignment). Delete both, replace with <gap>.

STEP 3 — VERIFY: Before outputting, check:
  - Does your output target contain FEWER words than the original?
    (If not, you forgot to delete from the target!)
  - Does the number of <gap> in source equal the number in target?
  - Are all remaining words copied exactly from the original?
  If any check fails, fix it before outputting.

{instruction}

Rules:
- You MUST delete words from BOTH source AND target — if you can't find
  the English words to delete, pick different source tokens where the
  alignment is clearer
- The output target must ONLY contain words from the original target
  (plus <gap>) — no new words
- The number of <gap> tokens in source MUST EQUAL the number in target
- Keep all remaining text EXACTLY as-is — same spelling, same order
- Ensure <gap> has spaces around it (not attached to punctuation)
- Never output two adjacent <gap> tokens — merge into one <gap>

Output a single JSON: {"src": "...", "tgt": "...",
  "alignment_used": "brief note of what you deleted and why"}

Example:
Source: KIŠIB šu-{d}EN.LÍL DUMU šu-ku-bi-im 10 ma-na KÙ.BABBAR
Target: Seal of Šu-Illil son of Šu-Kūbum, 10 minas of silver
Step 1 — KIŠIB=Seal, šu-{d}EN.LÍL=Šu-Illil, DUMU=son of,
         šu-ku-bi-im=Šu-Kūbum, 10=10, ma-na=minas, KÙ.BABBAR=silver
Step 2 — Delete šu-ku-bi-im from source → delete "Šu-Kūbum" from target
Step 3 — Target has 10 words, output has 9 words + 1 gap ✓.
         Gap count: 1=1 ✓
Output: {"src": "KIŠIB šu-{d}EN.LÍL DUMU <gap> 10 ma-na KÙ.BABBAR",
  "tgt": "Seal of Šu-Illil son of <gap> , 10 minas of silver",
  "alignment_used": "deleted šu-ku-bi-im=Šu-Kūbum"}

Example — two gaps:
Source: um-ma a-šùr-i-dí-ma a-na {m}PUZUR₄-a-šùr qí-bi₄-ma
Target: Thus says Aššur-idī: say to Puzur-Aššur
Step 1 — um-ma=Thus, a-šùr-i-dí-ma=says Aššur-idī, a-na=to,
         PUZUR₄-a-šùr=Puzur-Aššur, qí-bi₄-ma=say
Step 2 — Delete um-ma a-šùr-i-dí-ma → delete "Thus says Aššur-idī:".
         Delete qí-bi₄-ma → delete "say"
Step 3 — Target has 7 words, output has 2 words + 2 gaps ✓.
         Gap count: 2=2 ✓
Output: {"src": "<gap> a-na {m}PUZUR₄-a-šùr <gap>",
  "tgt": "<gap> to Puzur-Aššur <gap>",
  "alignment_used": "deleted um-ma a-šùr-i-dí-ma=Thus says Aššur-idī
    AND qí-bi₄-ma=say"}

Now do this pair:
Source: {src}
Target: {tgt}
Known alignments (use these as HINTS for Step 1 — but the target text
is the ground truth; only delete words that actually appear in the
target above, never insert glossary translations): {glossary}
```

The hardest part of gap generation is alignment — knowing which Akkadian tokens correspond to which English words. Without guidance, Gemini would guess and often get it wrong. The `{glossary}` solves this by providing known alignments for each pair upfront. For example, if the source contains `KÙ.BABBAR`, the glossary tells Gemini that this Sumerogram means "silver" — so when it deletes `KÙ.BABBAR` from the source, it knows to delete "silver" from the target. The glossary was built dynamically for each pair — scanning the source tokens and looking up only the ones present in that specific pair against three lookup tables: Sumerogram→English (`KÙ.BABBAR=silver`, `DUMU=son`), name translation memory (`A-šùr-na-da=Aššur-nādā`), and eBL dictionary entries.

Two levels of gap damage were generated, matching the real competition distribution:

**Whole-token gaps** — entire words replaced with `<gap>`, distributed across 11 variants: single middle gap (25%), edge gaps (11%), multi-gap scattered damage (64%). Example:

> **Input src:** `KIŠIB šu-{d}EN.LÍL DUMU šu-ku-bi-im 10 ma-na KÙ.BABBAR`
> **Input tgt:** *"Seal of Šu-Illil son of Šu-Kūbum, 10 minas of silver"*
>
> **Output src:** `KIŠIB šu-{d}EN.LÍL DUMU <gap> 10 ma-na KÙ.BABBAR`
> **Output tgt:** *"Seal of Šu-Illil son of <gap>, 10 minas of silver"*

**Mid-word gaps** — syllables removed within hyphenated words, matching 59% of competition test gaps. Three attachment types: word-end `ta-aš-<gap>` (49%), word-start `<gap>-ra-ni` (42%), mid-word `ta-<gap>-ni` (9%). Example:

> **Input src:** `ta-aš-pu-ra-ni` **Input tgt:** *"you wrote to me"*
> **Output src:** `ta-aš-<gap>` **Output tgt:** *"you <gap> to me"*

**Entity-swap augmentation** (~298K pairs generated, ~40K used) — Names, commodities, and numbers in existing pairs were programmatically swapped with alternatives from the OA onomasticon and lexicon. The swap function finds Akkadian name tokens in the source (using a pre-built index from the name translation memory), locates the corresponding English name in the target, and replaces both simultaneously:

```python
# Find Akkadian name in source → look up English equivalent → swap both
for akk_token in src_tokens:
    if akk_token in name_src_index:
        eng_name = name_src_index[akk_token]
        if eng_name in tgt:
            # Pick random replacement from candidates
            new_akk, new_eng = random.choice(name_candidates)
            new_src = src.replace(akk_token, new_akk)
            new_tgt = tgt.replace(eng_name, new_eng)
```

Example:

> | | src | tgt |
> |---|---|---|
> | **Original** | `2 ma-na KÙ.BABBAR ša Puzur₄-a-šùr` | *"2 minas of silver belonging to Puzur-Aššur"* |
> | **Swapped** | `2 ma-na KÙ.BABBAR ša en-na-nim` | *"2 minas of silver belonging to Ennanim"* |

## Training

All four ensemble models use **google/byt5-base** trained in **FP32**. ByT5's byte-level tokenisation avoids OOV issues with Akkadian diacritics (`š ṣ ṭ ḫ á à`) — these are multi-byte UTF-8 characters that subword tokenizers can split unpredictably. FP32 precision matters at the byte level.

The models are deliberately diverse — different data compositions, learning rates, and augmentation strategies to create complementary specialists.

### Data Provenance Per Model

The extracted and cleaned data was combined differently for each model:

| | **A — Soup** | **B — Gap stage 2** | **C — Gap joint** | **D — Distribution** |
|---|---|---|---|---|
| **What's different** | No augmentation, weight-averaged 3 checkpoints | 2-stage: base training → gap fine-tune | Joint training with gaps + expanded data + higher LR | 80/20 real/augmented blend matching test distribution |
| **Training** | Single stage, LR 1e-4 | 2-stage: base → gap fine-tune, LR 1e-4 | Single stage, LR 3e-4, encoder freeze after epoch 5 | Single stage, LR 1e-4 |
| **Checkpoint** | Soup of epochs 3/5/6 | Gap epoch 3 | Gap epoch 3 | Epoch 6 |
| | | | | |
| **Data** | | | | |
| Competition (doc + sent) | 1,561 + ~5.4K (3×) | same | 1,561 + ~5.4K (3×) | 1,561 + ~5.4K (3×) |
| PDF extraction | 94,893 | same | 94,893 | 95,399 |
| AKT volumes | 15,534 | same | 15,534 (2×) | 15,576 |
| OARE sentences | 6,823 | same | 6,823 (2×) | 6,972 |
| Larsen, secondary, eSAD, Prague | ~3,800 | same | ~3,800 (2×) | ~3,849 |
| Gemini Vision extracted PDFs | — | — | 30,793 (2×) | 30,793 |
| Synthetic gap pairs | — | ~124K | ~124K | — |
| Entity-swap augmented (score-4) | — | — | — | 40,299 |
| **Approx total (bidirectional)** | **~254K** | **~502K** | **~690K** | **~400K** |

All models trained with the standard HuggingFace [Seq2SeqTrainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer): google/byt5-base, FP32, Adafactor, label smoothing 0.2, bidirectional training (each pair trained in both directions — Akkadian→English and English→Akkadian — to double effective data), dynamic batching.

### Validation

90/10 random hold-out split, chrF evaluated per epoch. Checkpoint selection was ultimately LB-driven — internal eval chrF consistently diverged from LB score (e.g. eval chrF improved +4 points across epochs with no LB gain).

### Hardware

All training ran on an NVIDIA DGX Spark (Grace Blackwell GB10, unified memory). Each model trained in ~6-10 hours depending on data size.

## Inference & Ensemble

Each model generates multiple candidate translations via beam search. Rather than taking the single highest-scoring beam, MBR (Minimum Bayes Risk) decoding selects the candidate that is most similar to all other candidates — the intuition being that if multiple beams agree on a translation, it's more likely to be correct:

$$\hat{y} = \arg\max_{y \in \mathcal{C}} \sum_{y' \in \mathcal{C}} u(y, y')$$

where $\mathcal{C}$ is the set of candidates and $u(y, y')$ is a utility function scoring how similar two candidates are. Each model generates candidates with 12 beams, returning the top 4 per model (16 candidates total across 4 models). Length penalty is adaptive: 1.05 for short sources (≤20 tokens), scaling to 1.3 for long sources (>40 tokens), linearly interpolated between. A local sweep showed higher LP gained +1.13 GeoMean, but pushing further (fixed 1.3, adaptive 1.2/1.5) hurt on LB — over-generation penalised short test sentences.

The final score for each candidate combines three components:

$$u(y) = \alpha \cdot \text{chrF}_{\text{consensus}}(y) + \beta \cdot \text{lexicon}(y) + \gamma \cdot \text{SG}(y, x)$$

- **chrF consensus** — character n-gram F-score measuring agreement between candidates
- **Lexicon coverage** — fraction of output words found in a known Old Assyrian word list
- **SG (source-grounded)** — faithfulness score checking the candidate against verifiable features in the source

The base weights ($\alpha=0.35, \beta=0.10, \gamma=0.55$) shift dynamically based on how much of the source is verifiable.

Before MBR runs, a **translation memory** (exact-match lookup built from `train.csv`) intercepts any test source that exactly matches a training source — these bypass the model entirely and return the known gold translation. This is a safe, zero-risk boost for the ~5% of test sources that overlap with training data.

After MBR selects a winner, a 9-step **post-processing pipeline** cleans the output:
1. Character normalisation (`ḫ`→`h`, subscripts→digits)
2. Gap normalisation (stray `x`, `...` → `<gap>`, collapse adjacent gaps)
3. Annotation removal (`fem.`, `sing.`, `pl.`, `(?)`)
4. OA-specific expansions (`-gold`→`pašallum gold`, Roman months→integers)
5. Fraction conversion (decimals→Unicode `½ ⅓ ⅔`, slash fractions→Unicode)
6. Forbidden character removal (half-brackets, angle brackets, smart quotes→straight)
7. Repetition removal (repeated words and phrases)
8. Sumerogram expansion (divine name Sumerograms leaked into target)
9. Spacing cleanup and final gap collapse

After post-processing, a 3-stage **name repair** pipeline corrects names using a ~6,800-entry translation memory: (1) determinative-based repair matches `{d}`, `{m}`, `{f}` tokens in the source to expected name translations, (2) source-guided repair handles names without determinatives, (3) competition-specific fallback fixes remaining ASCII-only names.

### Source-Grounded Scoring

The SG score scans the source for verifiable features — numbers, Sumerograms, divine names, gaps — and checks whether each candidate faithfully translates them. Each feature's weight is proportional to how many characters of the candidate it covers, so features that account for more of the output have more influence:

$$w_i = \frac{\text{chars}_i}{\sum_j \text{chars}_j}, \quad \text{SG}(y, x) = \sum_i w_i \cdot \text{score}_i(y, x)$$

For content features, charsᵢ is computed automatically from the source — e.g. `{d}UTU` maps to "Šamaš" (5 chars), so in a 46-char candidate the divine name feature gets weight 5/46 = 11%. Always-on features (repetition detection, echo detection, length ratio, Sumerogram residue, content grounding, document type) use fixed fractions. No weights are hand-tuned.

### Ensemble Example

Here are the actual outputs from all 4 models on an Old Assyrian debt contract — 5 minas of silver with an interest rate, a `<gap>` where the tablet is damaged, and a date clause:

**Source:** `5 ma-na KÙ.BABBAR KI a-šur-ta-ak-lá-ku iš-tù ḫa-mu-uš-tim ša a-mur-IŠTAR ú a-šur-bé-el-a-wa-tim <gap>-i ITU.KAM té-i-na-tim li-mu-um a-mur-a-šur DUMU kà-ri-a 1½ GÍN.TA i-na ITU.1.KAM-im a-na ma-na-im ú-ṣa-áb`

| Model | Top candidate |
|-------|---------------|
| **A (soup)** | 5 minas of silver **to the debit of** Aššur-taklāku. Reckoned from the week of Amur-Ištar and Aššur-bēl-awātim, month Te'inātum, eponymy of Amur-Aššur son of Karriya, he will add 1½ shekels per mina per month. |
| **B (gap)** | 5 minas of silver **is owed by** Aššur-taklāku. From the week of Amur-Ištar and Aššur-bēl-awātim, **`<gap>`** Month Teinātum, eponymy Amur-Aššur son of Karriya, he will add 1½ shekels per month per mina. |
| **C (joint)** | 5 minas of silver **is with** Aššur-taklāku. Reckoned from the week of Amur-Ištar and Aššur-bēl-awātim, **`<gap>`** month Te'inātum, eponymy Amur-Aššur son of Karriya, he will add 1½ shekel per month per mina. |
| **D (dist)** | 5 minas of silver **is owed by** Aššur-taklāku. Reckoned from the week of Amur-Ištar and Aššur-bēl-awātim, month Te'inātum, eponymy Amur-Aššur son of Karriya, he will add interest at the rate 1½ shekel per month per mina. |
| **Reference** | 5 minas of silver is owed by Aššur-taklāku. Reckoned from the week of Amur-Ištar and Aššur-bēl-awātim `<gap>` month Teʾinātum, eponymy of Amur-Aššur son of Karriya, he will pay interest at the rate of 1½ shekel per month per mina. |

The models split on two axes: **gap fidelity** (B and C include `<gap>`, A and D drop it) and **phrasing** ("to the debit of" vs "is owed by" vs "is with"). All correctly translate the Sumerogram (`KÙ.BABBAR`→"silver"), numbers (`5` and `1½`), and names.

SG-MBR scoring on representative candidates from the full pool of 16:

| Candidate | chrF consensus | SG score | Gap (source has 1) | Sumerogram (`KÙ.BABBAR`→silver) | Numbers (`5`, `1½`) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| A1 — no gap | 0.839 | 0.857 | **0/1 — missing** | 1.0 | 2/2 |
| B3 — with gap | 0.871 | **0.869** | **1/1** | 1.0 | 2/2 |
| C3 — with gap | 0.874 | **0.869** | **1/1** | 1.0 | 2/2 |
| **D1 — no gap** | **0.875** | 0.858 | **0/1 — missing** | 1.0 | 2/2 |

**D1 has the highest chrF consensus** (0.875) — its phrasing ("is owed by ... Reckoned from") is the most common across all 16 candidates, so pure chrF MBR would select it. But D1 drops the `<gap>` that exists in the source, so its SG score is lower (0.858 vs 0.869). The gap match feature checks: the source contains `<gap>-i`, so candidates that include `<gap>` score higher.

With the combined weighting (α=0.49 chrF + γ=0.44 SG for this source), C3 overtakes D1 — selecting a translation that faithfully reflects the damaged tablet over one with marginally more popular phrasing. This is the core value of source-grounded MBR: using verifiable source features to break ties that chrF consensus alone cannot resolve.

Switching from a single model to source-grounded MBR with diverse models contributed +2.58 on private LB (37.6→40.1). On public LB, multi-checkpoint MBR of a single model scored higher (39.6) but this didn't generalise — cross-model SG-MBR scored 39.2 public but gained +1.78 private over the best single-model private score (38.4).

## Score Progression

| Submission | Setup | Public | Private | Notes |
|-----------|-------|--------|---------|-------|
| Model A alone (soup) | 1 model | 37.7 | 37.6 | Weight average of 3 checkpoints, rock solid |
| Model B alone | 1 model | 38.9 | 37.3 | Gap training, public overfit |
| Model C (3 checkpoints) | 1 model, chrF MBR | 39.0 | 37.9 | Higher LR + expanded data |
| Model B (3 checkpoints) | 1 model, chrF MBR | 39.6 | 38.4 | Checkpoint MBR inflates public |
| Model B + D | 2 models, chrF MBR | 39.4–39.8 | 38.4–38.6 | Diversity without hurting |
| **A + B + C + D SG-MBR** | **4 models, SG-MBR** | **39.2** | **40.1** | **Diverse models generalise** |

Multi-checkpoint MBR of one model inflated on public LB (39.6→38.4, −1.2). Cross-model diversity with SG-MBR gained on private (39.2→40.1, +0.98). Diverse training runs generalise better than diverse checkpoints of the same model.

## What Didn't Work

### ByT5-large

Didn't outperform base despite more capacity — likely memorised noise in the training data while the smaller model was forced to generalise. Only one training run was attempted with the same data mix as the base models. With more time, cleaning the noisiest extracted pairs and retraining may have unlocked the larger model's potential — the 3rd place team used ByT5-Large and XL successfully with cleaner synthetic data.

### DAPT (Unsupervised Pre-Training)

One attempt at Domain-Adaptive Pre-Training was made on ~820K lines of raw Akkadian. The encoder learned better representations (probing showed 61% name copying on novel names vs 29% baseline), but the decoder's cross-attention was still tuned to the vanilla ByT5 encoder. This misalignment caused repetition loops at inference, scoring 34.6 vs 38.5 baseline (−3.9 points). With more GPU time, approaches like cross-attention warmup schedules or the 3rd place team's supervised CPT strategy (using seq2seq synthetic drills rather than unsupervised text) may have resolved the misalignment — but there wasn't enough compute budget to iterate further.

### DPO (Direct Preference Optimisation)

Eval metrics improved but LB score dropped — the preference optimisation overfitted to training gold n-grams rather than learning generalisable translation quality.

### Sampling MBR

Sampling at temperature 0.7 with MBR-8 gained +2.95 GeoMean on 200 local samples but **dropped 2.7 points on LB** — local eval on seen data was systematically misleading.

## Key Lessons

1. **Data extraction was the biggest lever.** publications.csv contained far more training data than train.csv. Extracting and cleaning it made the most difference.

2. **Cross-model diversity > checkpoint diversity.** Multi-checkpoint MBR of one model inflated on public LB (39.6 public → 38.4 private, −1.2). Four models with different data/LR/augmentation held up (39.2 → 40.1, +0.98). Diverse training runs generalise; diverse checkpoints of the same run don't.

3. **Systematic OCR repair goes a long way.** After 30+ cleaning passes, the host OCR data was good enough that re-extracting with Gemini Vision didn't improve LB scores further — though Vision was essential for PDFs without a text layer.

4. **Data quality > model size.** ByT5-base outperformed ByT5-large. Clean your data before scaling your model.

5. **FP32 for byte-level models.** ByT5 operates at the byte level — precision matters when distinguishing `š` from `s` or `ḫ` from `h`. BF16's reduced mantissa blurs these distinctions. All training used FP32.

## Sources

- **Submission notebook:** https://www.kaggle.com/code/eleftheria14/3-model-source-grounded-mbr-chrf-35-10-55?scriptVersionId=304619615 (notebook title says "3-model" — it was later expanded to 4 models)
- **ByT5:** Xue et al., [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626) (2022)
- **HuggingFace Seq2SeqTrainer:** Wolf et al., [Transformers: State-of-the-Art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6/) (2020)
- **MBR decoding:** Eikema & Aziz, [Is MAP Decoding All You Need?](https://arxiv.org/abs/2005.10283) (2020) — our implementation uses chrF-based consensus MBR over beam search candidates rather than the sampling approach from this paper
- **Gemini:** Google, [Gemini 2.0 Flash / Flash Lite](https://ai.google.dev/gemini-api/docs/models) — used for PDF extraction, Vision OCR, sentence splitting, and gap augmentation
- **Claude Code** with Claude Opus 4.6 (Anthropic) — used for data cleaning script development and quality judging
- **Key publication sources for extracted data:** Larsen 2002 (PIHANS 96), AKT volumes 1-13, Dercksen 2004/2008 (PIHANS 98/111), Michel 2003 (PIHANS 97), Prague ICK 4, secondary_sources.csv, eSAD
