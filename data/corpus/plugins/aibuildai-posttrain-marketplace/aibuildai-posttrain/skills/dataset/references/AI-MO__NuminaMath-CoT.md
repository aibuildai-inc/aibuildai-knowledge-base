# AI-MO/NuminaMath-CoT

859,594 competition and exam math problems paired with chain-of-thought solutions, mixing Chinese K-12 exercises, US/international olympiad problems, and synthetic problems, in a single `train`/`test` split.

**AI-MO/NuminaMath-CoT** is Numina's release of the dataset behind its report "NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions" [1], built by acquiring problems from exam PDFs and math forums, then OCR-ing, segmenting into problem-solution pairs, translating to English, and realigning the solutions into a uniform chain-of-thought (CoT) format with the final answer boxed [1]. Each row carries a `source` tag naming which of nine pools it came from (`aops_forum`, `amc_aime`, `cn_k12`, `gsm8k`, `math`, `olympiads`, `orca_math`, `synthetic_amc`, `synthetic_math`) [2]. It lives at https://huggingface.co/datasets/AI-MO/NuminaMath-CoT . **The pool includes AoPS-forum, olympiad, and AMC/AIME-derived problems; the report's own decontamination only removed overlap with the MATH, GSM8K, AIME 2024, and AMC 2023 evaluation sets [1], so overlap with evaluations released after that decontamination pass - AIME 2025 in particular - is not addressed by any source read for this card, and a reader scoring a model against a post-2024 competition benchmark should screen this pool against that benchmark first.**

**Use it for**: reasoning-trace SFT - each row's `messages` field is a two-turn user/assistant chat pair holding the problem and its CoT solution, so it maps directly to the SFT method card's chat format. Hold out `test` (100 rows) before training. Not preference data - there is no rejected side.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [3]. No catch beyond the standard Apache-2.0 grant is stated by any source read for this card.

**Shape**: 859,594 rows, one config (`default`), two splits: `train` 859,494 / `test` 100 [4].

**Hold out**: `test` (100 rows), a same-schema sample across the same nine source pools [2][5]; no source states how these 100 rows were selected. Beyond that split, screen against post-2024 competition benchmarks per the restriction above - the card carries no row-level flag for that risk beyond the pool composition itself.

**Origin**: built and released by Numina (an open AI4Maths initiative), a collaboration credited on the report to Hugging Face, MIT, Mistral AI, Peking University, and Answer AI staff [1]. Solutions are machine-processed (OCR, translation, and CoT realignment by GPT-4/GPT-4o) from human-authored source material, not human-written from scratch [1]. Hub API at the check date: `downloads` 74,557, `downloadsAllTime` 394,390, `likes` 595 [3].

**Trained-on-by**: the report's own NuminaMath-7B and NuminaMath-72B models, fine-tuned on this dataset (Stage 1 of a two-stage recipe) and its tool-integrated-reasoning extension (Stage 2), which won the 1st AIMO Progress Prize [1]. Table 3's Stage-1 (CoT-only, trained on this release) NuminaMath-7B is not uniformly ahead of the baselines it is compared against in that same table: it leads on MATH (55.8% 0-shot, against 49.6% for Qwen2-7B-Instruct, the next-best baseline shown), but it is the lowest of the seven models on GSM8k 0-shot (76.3%, against 79.6%-88.2% for the other six models), and it ties Llama3-8B-Instruct for the lowest AIME 2024 0-shot score (0/30, against 1/30 for the other four models) [1]. No source read for this card documents a third-party model trained on this exact release; the derived `AI-MO/NuminaMath-TIR` dataset (below) is the report's own tool-integrated-reasoning subset used for the Stage-2 model in that same win.

**Introduced by**: [1] (Li et al., the report "NuminaMath").

## Shape

Rows and splits, from the datasets-server size endpoint [4]:

| split | rows |
| --- | --- |
| `train` | 859,494 |
| `test` | 100 |
| total | 859,594 |

One config, `default`, four columns (datasets-server `/info`, matching the README's `dataset_info` block) [5][2]:

| column | dtype |
| --- | --- |
| `source` | string |
| `problem` | string |
| `solution` | string |
| `messages` | list<struct<content: string, role: string>> |

`source` values and row counts, from the README's own table (its stated total of 859,608 is 14 rows over the 859,594 actually served) [2]:

| source | rows (README table) |
| --- | --- |
| cn_k12 | 276,591 |
| synthetic_math | 167,895 |
| orca_math | 153,334 |
| olympiads | 150,581 |
| synthetic_amc | 62,111 |
| aops_forum | 30,201 |
| math | 7,478 |
| gsm8k | 7,345 |
| amc_aime | 4,072 |
| **total (README)** | **859,608** |

Reading all 100 served `test` rows live gives this split's own breakdown: cn_k12 35, synthetic_math 21, orca_math 20, olympiads 13, aops_forum 3, gsm8k 3, synthetic_amc 3, amc_aime 1, math 1 [6] - proportionally close to the full-dataset table above, consistent with `test` being a same-mixture sample.

Sizes, from the datasets-server size endpoint [4]: 1,234,351,634 bytes as Parquet (matching the README's `download_size`), 2,454,252,909 bytes decoded in memory. No source read for this card states sequence-length or token-count statistics for this release; the report does say that about 40% of the combined NuminaMath problem pool is proof-based, particularly from the IMO [1], but gives no per-release token counts.

## Quality

- The report states two known contamination checks were applied to the training data before release: a 10-gram exact-string-match filter against MATH, GSM8K, AIME 2024, and AMC 2023, and a Mistral-embedding nearest-neighbor filter (distance threshold 0.15) against AIME/AMC problems specifically, because the `olympiads` subset draws from globally-sourced contest problems that can be translated versions of AMC/AIME items [1]. Both checks target those four named evaluation sets only; no source states a check against any other benchmark, including AIME 2025.
- The report gives one worked example of the embedding filter catching a paraphrased AIME 2001 problem that also appears (reworded) in the MATH test set [1].
- The report lists several unresolved quality issues in its own future-work section: CoT solutions for the hardest problems "may lack sufficient detail," about 40% of problems are proof-based and not yet translated into formal mathematical proofs, and the synthetic subsets (`synthetic_math`, `synthetic_amc`) are generated at nonzero temperature (0.8) and are "challenging to validate" for correctness [1].
- No source read for this card states a measured contamination rate, duplicate rate, or human-annotation-agreement figure; the checks above are process descriptions, not measured rates.

## Load it

Train on `train`, hold out `test`, and pin the revision this card's `sha` and licence facts were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-11-25) [3]. The row counts in the comments below come from the datasets-server endpoints, which take no revision parameter, so they are live figures as of the check date rather than values this `sha` pin guarantees [4]:

```python
import datasets

REV = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"  # main at the check date
train = datasets.load_dataset("AI-MO/NuminaMath-CoT", revision=REV, split="train")  # 859,494 rows
test = datasets.load_dataset("AI-MO/NuminaMath-CoT", revision=REV, split="test")    # 100 rows - hold out
```

**Trap**: the `messages` field is the two-turn chat pair to train on directly; `problem` and `solution` are the same content unpacked into separate raw-text columns, so an SFT run that concatenates `problem` + `solution` by hand instead of using `messages` duplicates the same training signal under a different key rather than adding new data.

## Neighbors

Two releases from the same builder overlap this one; a row read from each, live, shows how [7][8][9].

- `AI-MO/NuminaMath-1.5` - the successor release, 896,215 rows in a single split (no held-out `test`) [8], adding `answer`, `problem_type`, and `question_type` metadata columns to every row, keeping the original `olympiads` pool unchanged (197,084 rows in the 1.5 breakdown) while adding a new, separate `olympiads_ref` pool (3,638 rows) manually parsed and verified from official national-olympiad websites to address parsing issues the builder found in the original `olympiads` subset, adding manually curated `cn_contest`, `inequalities`, and `number_theory` pools, and dropping the `synthetic_amc` pool entirely because an ablation showed it hurt performance [9]. It has no `test` split, so a reader who needs a held-out sample from Numina's own data must build one, or continue to use this release's `test` for that purpose.
- `AI-MO/NuminaMath-TIR` - a 72,540-row subset of this release's problems (72,441 train / 99 test) [7], re-annotated with tool-integrated-reasoning (TIR) trajectories - interleaved CoT text and executed Python code - instead of pure-text CoT solutions; its own card states it drew "approximately 70k problems from the NuminaMath-CoT dataset, focusing on those with numerical outputs" [10]. Its schema drops the `source` column and keeps `problem`, `solution`, `messages`, so a `messages`-only collator built for this release also runs on it, but its `messages` hold TIR text-plus-code turns, not plain CoT prose.

This corpus card treats the present release, `AI-MO/NuminaMath-CoT`, as the default: it is the plain-CoT, largest, and most-downloaded of the three at the check date (74,557 downloads versus 11,294 for `NuminaMath-1.5` and 6,094 for `NuminaMath-TIR`) [11]. Pick `NuminaMath-1.5` for its extra metadata and fixed `olympiads` parsing if a held-out split can be built separately; pick `NuminaMath-TIR` only when training a tool-use model specifically.

## A row

One config and one schema, shared by both splits - a `test`-split row was read to confirm the same four columns and dtypes appear there [2][5][6]. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [12]:

```json
{
  "source": "synthetic_math",
  "problem": "Consider the terms of an arithmetic sequence: $-\\frac{1}{3}, y+2, 4y, \\ldots$. Solve for $y$.",
  "solution": "For an arithmetic sequence, the difference between consecutive terms must be equal. Therefore, we can set up the following equations based on the sequence given:\n\\[ (y + 2) - \\left(-\\frac{1}{3}\\right) = 4y - (y+2) \\]\n\nSimplify and solve these equations:\n\\[ y + 2 + \\frac{1}{3} = 4y - y - 2 \\]\n\\[ y + \\frac{7}{3} = 3y - 2 \\]\n\\[ \\frac{7}{3} + 2 = 3y - y \\]\n\\[ \\frac{13}{3} = 2y \\]\n\\[ y = \\frac{13}{6} \\]\n\nThus, the value of $y$ that satisfies the given arithmetic sequence is $\\boxed{\\frac{13}{6}}$.",
  "messages": [
    {"content": "Consider the terms of an arithmetic sequence: $-\\frac{1}{3}, y+2, 4y, \\ldots$. Solve for $y$.", "role": "user"},
    {"content": "For an arithmetic sequence, the difference between consecutive terms must be equal. [...] Thus, the value of $y$ that satisfies the given arithmetic sequence is $\\boxed{\\frac{13}{6}}$.", "role": "assistant"}
  ]
}
```

## Where it came from

Numina built the dataset from nine pools, each collected by its own method [1]: `math` and `gsm8k` solutions were reformatted from the original MATH and GSM8K training data using GPT-4, following prior work's CoT-reformatting recommendations; `orca_math` answers were extracted from the existing Orca-Math dataset by regex and boxed; `amc_aime` problems were collected from the AoPS wiki, matched to community-proposed solutions, decontaminated against MATH (retaining roughly 4,300 of an initial ~6,500 problems), and realigned into CoT by GPT-4o; `aops_forum` problems were crawled from the AoPS Contest Collection page, paired with the best community reply found, and realigned into CoT by GPT-4o; `cn_k12` exercises were OCR'd from Chinese exam-paper PDFs, then translated and realigned by GPT-4o; `synthetic_math` and `synthetic_amc` were generated by prompting GPT-4 with seed problems at temperature 0.8; and `olympiads` problems (152K pairs) were collected from international and national contest shortlists, problem-solving forums, and olympiad books, segmented by contest-specific regex or manual OCR, then run through the same GPT-4o pipeline used for the other OCR'd and web-collected pools. The report names this shared GPT-4o step once for `amc_aime`, `aops_forum`, and `cn_k12`, then gives it in full detail in a later section covering translation, CoT realignment, and boxing the final answer - it is one processing step described at two levels of detail, not a second pass, and `olympiads` is the one pool for which that detailed section is its first mention [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Li, Beeching, Tunstall, Lipkin, Soletskyi, Huang, Rasul, Yu, Jiang, Shen, Qin, Dong, Zhou, Fleureau, Lample, and Polu, "NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions," Numina report, 2024-07-22. https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf (fetched via the raw file at https://raw.githubusercontent.com/project-numina/aimo-progress-prize/main/report/numina_dataset.pdf) - Sections 1, 3, 3.1-3.4, 4, and 5. Fetched 2026-08-11.

[2] AI-MO/NuminaMath-CoT dataset card (README), fetched at the revision pinned in Load it. https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/9d8d210c9f6a36c8f3cd84045668c9b7800ef517/README.md - `dataset_info` schema, source-breakdown table, licence; confirmed byte-identical to the `main` copy read earlier. Fetched 2026-08-11.

[3] Hugging Face Hub API record for AI-MO/NuminaMath-CoT. https://huggingface.co/api/datasets/AI-MO/NuminaMath-CoT?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=AI-MO%2FNuminaMath-CoT - this endpoint takes no revision parameter, so its row counts and byte sizes are live, not pinned; they are not covered by the `sha` pinned in Load it. Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=AI-MO%2FNuminaMath-CoT - no revision parameter; live, not pinned, and not covered by the `sha` pinned in Load it. Fetched 2026-08-11.

[6] datasets-server rows endpoint, full 100-row `test` split. https://datasets-server.huggingface.co/rows?dataset=AI-MO%2FNuminaMath-CoT&config=default&split=test&offset=0&length=100 - read all 100 rows to build the split's own source breakdown; no revision parameter, so this is live, not pinned, and not covered by the `sha` pinned in Load it. Fetched 2026-08-11.

[7] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=AI-MO%2FNuminaMath-TIR - this endpoint takes no revision parameter, so this count is live, not pinned. Fetched 2026-08-11.

[8] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=AI-MO%2FNuminaMath-1.5 - live, not pinned. Fetched 2026-08-11.

[9] AI-MO/NuminaMath-1.5 dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/raw/main/README.md - new metadata columns, `olympiads` re-parsing, new pools, `synthetic_amc` removal. Fetched 2026-08-11.

[10] AI-MO/NuminaMath-TIR dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-TIR/raw/main/README.md - subset size and TIR generation description. Fetched 2026-08-11.

[11] Hugging Face Hub API records for all three releases (author listing). https://huggingface.co/api/datasets?author=AI-MO&full=true - `downloads` figures compared across `NuminaMath-CoT`, `NuminaMath-1.5`, `NuminaMath-TIR`. Fetched 2026-08-11.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2FNuminaMath-CoT&config=default&split=train - no revision parameter; live, not pinned, and not covered by the `sha` pinned in Load it. Fetched 2026-08-11.

[13] The corpus screening row for `AI-MO/NuminaMath-CoT`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT, with `test` (100 rows) held out. The dataset's own report documents decontamination against MATH, GSM8K, AIME 2024, and AMC 2023 only [1]; the screening row flags a rule-risk specific to a later benchmark, AIME 2025, because this release's `aops_forum`, `olympiads`, and `amc_aime` pools are exactly the kind of competition-problem pool that a post-release benchmark like AIME 2025 could overlap - a risk this card's own sources do not resolve, so a reader running that benchmark should screen this pool against it before scoring.

### The screening row

The row's own note [13]: "860k human competition/exam problems (AoPS, cn_k12, olympiads) with CoT solutions; `train` safe, `test` is a 100-row held-out sample of the same mixture; its gsm8k subset is 7,345 rows, below the 7,473 train split." Its flag [13]: "rule-risk: aime2025 - This is the NuminaMath problem pool itself - AoPS, olympiads and amc_aime competition items - upstream of the ruled calibration case."
