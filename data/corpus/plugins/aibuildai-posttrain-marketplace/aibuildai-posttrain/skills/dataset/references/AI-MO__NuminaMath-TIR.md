# AI-MO/NuminaMath-TIR

72,540 GPT-4/GPT-4o-generated tool-integrated-reasoning (TIR) traces - interleaved chain-of-thought text and executed Python code - over roughly 70k competition-style math problems drawn from NuminaMath-CoT.

**AI-MO/NuminaMath-TIR** was built by the Numina project (Jia Li et al., with Hugging Face, MIT, Mistral AI, Peking University and Answer AI collaborators) by extracting a subset of value-output problems from NuminaMath-CoT and re-solving each with a pipeline that interleaves natural-language reasoning with executed Python code, discarding candidates whose final answer did not match the reference; the dataset card names the generating model as GPT-4 [1], while the origin paper's own construction section names the GPT-4o assistant API for sampling and GPT-4o as the answer-matching judge [2, Sec. 3.3]. **The parent NuminaMath-CoT pool this subset was drawn from includes 4,072 `amc_aime` and 30,201 `aops_forum` competition problems [3], and the paper's own decontamination pass (10-gram exact match plus a Mistral-embedding nearest-neighbor filter) was run only against MATH, GSM8K, AIME 2024 and AMC12 2023 [2] - a scope that predates AIME 2025 and does not name it, and the served TIR rows carry no per-example source label to isolate the AMC/AIME-derived subset (see Quality). Decontaminate against any AIME- or AMC-derived benchmark, including AIME 2025, before a scored run.** It lives at https://huggingface.co/datasets/AI-MO/NuminaMath-TIR .

**Use it for**: reasoning-trace SFT that teaches interleaved CoT-plus-code-execution answers - the `messages` column is a ready user/assistant pair per row, mapping directly to a chat-SFT format (the reasoning-trace SFT method card); the restriction above applies before evaluating against AIME/AMC-style benchmarks.

**Licence**: apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [4]. No further usage restriction is stated in the dataset card beyond the contamination caution above, which is this card's own reading of the paper's decontamination scope, not a licence term.

**Shape**: 72,540 rows in one config (`default`), split `train` 72,441 / `test` 99, three columns (`problem`, `solution`, `messages`) [5][6].

**Hold out**: `test` (99 rows) as the card's own split [5]. Beyond that split, no dataset-level filter isolates the AMC/AIME-derived rows described above; a reader targeting an AIME- or AMC-based benchmark has to decontaminate against it directly, since the served schema has no source column (see Quality).

**Origin**: built by Numina; problems come from NuminaMath-CoT's human-collected competition and exam sources, and every solution/reasoning trace is a model generation - the dataset card names GPT-4 [1], the origin paper names the GPT-4o assistant API and a GPT-4o judge [2, Sec. 3.3]. Hub API at the check date: `downloads` 6,645, `downloadsAllTime` 112,415, `likes` 156 [4][7].

**Trained-on-by**: Numina's own NuminaMath-7B-TIR (`deepseek-ai/deepseek-math-7b-base`) and NuminaMath-72B-TIR (`Qwen2-72B`) - the paper's Stage 2 fine-tunes on this synthetic TIR dataset, which won the 1st AIMO Progress Prize (7B model score 29/50 on the public and private test sets) [2][8]. The paper's own head-to-head on the 7B model shows the TIR data raises scores over CoT-only training on the same base model: MATH 55.8% (CoT) to 68.1% (TIR), GSM8K 76.3% (CoT) to 84.6% (TIR), AMC 2023 0-shot 11/40 (CoT) to 20/40 (TIR), AIME 2024 0-shot 0/30 (CoT) to 5/30 (TIR) [2, Table 3]. No other adopter was found in the sources checked.

**Introduced by**: [2] (Li et al., "NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions").

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 72,441 |
| `test` | 99 |
| total | 72,540 |

One config, `default`, with three columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `solution` | string |
| `messages` | list<struct<content: string, role: string>> |

`messages` holds exactly two turns per row in the ten rows sampled from `train` (offsets 0-9) and the one row sampled from `test` (offset 0): a `user` turn (the problem) and an `assistant` turn (the full TIR trace, identical in content to the `solution` field) [9][10]. The upstream NuminaMath-CoT pool this subset was drawn from carries a fourth column, `source`, naming which competition/exam pool each problem came from (`amc_aime`, `aops_forum`, `cn_k12`, etc.) [3][11]; that column is not present here, so the served rows carry no per-example provenance label.

Sizes (datasets-server `/size`) [5]: 147,557,990 bytes of original Parquet download, 323,813,759 bytes decoded in memory. No source states sequence-length or token statistics for this release; the origin paper reports token statistics only for the parent NuminaMath-CoT olympiad subset, not for TIR [2].

## Quality

- The TIR pipeline sampled a candidate solution per problem from the GPT-4o assistant API at temperature 0.8, kept only those whose final answer matched the reference (exact match for integer outputs, GPT-4o as judge for other expressions), and repeated the sampling on the remaining unsolved (negative) problems [2, Sec. 3.3]; the dataset card states this repetition ran three times and names the generating model as GPT-4 rather than GPT-4o [1]. No measured final pass rate is stated by either source.
- The parent NuminaMath-CoT pool underwent a two-step decontamination before this subset was drawn: exact 10-gram matching against MATH, GSM8K, AIME 2024 and AMC12 2023, plus a Mistral-embedding nearest-neighbor filter (empirically set at distance < 0.15) targeted specifically at AMC/AIME-style paraphrases, because the paper notes MATH's own test set already contains reworded AIME problems [2, Sec. 3.4]. That decontamination scope names AIME 2024 and AMC12 2023 only; it does not name AIME 2025, and the dataset's last Hub update (2024-11-25) [4] predates the AIME 2025 competition.
- The NuminaMath-CoT source breakdown table lists 4,072 `amc_aime` and 30,201 `aops_forum` problems among its roughly 860k total [3]; TIR's ~70k-problem subset was drawn from this pool without regard to source [2, Sec. 3.3], and the served TIR rows carry no `source` column to identify which rows came from those two pools (see Shape). Recovering that label requires joining TIR's `problem` text back to NuminaMath-CoT's `problem` + `source` columns; both datasets' served rows carry the `problem` field as the join key, but no source states how many TIR rows would match an `amc_aime` or `aops_forum` row under that join, and none is invented here.
- No source states a duplicate-row rate or annotator-agreement figure for this dataset.

## Load it

Train on `train`, hold out `test`, and pin the revision this card's numbers were read at (the Hub API's `sha`, matching the shortlist commit; the repo was last modified 2024-11-25) [4]:

```python
import datasets

REV = "77a91d7b7a1a98ac4b1beb7d86c09d156b935dcd"  # main at the check date
train = datasets.load_dataset("AI-MO/NuminaMath-TIR", revision=REV, split="train")  # 72,441 rows
test = datasets.load_dataset("AI-MO/NuminaMath-TIR", revision=REV, split="test")    # 99 rows - hold out
```

**Trap**: `test` here is not a competition benchmark - it is a 99-row held-out sample of the same TIR pool as `train` [5]. It does not correspond to any of the paper's own evaluation sets (MATH, GSM8K, AMC 2023, AIME 2024), so scoring against it does not substitute for a decontaminated AIME/AMC evaluation.

## Neighbors

- `AI-MO/NuminaMath-CoT` - the ~860k-row parent pool this ~70k-problem TIR subset was drawn from; chain-of-thought solutions only, no code execution, and it carries the per-row `source` column that TIR drops [2][3]. Fetching row 0 of its `train` split confirms the schema: `source`, `problem`, `solution`, `messages` [11].
- `AI-MO/NuminaMath-1.5` - the CoT pool's successor (896,215 rows), adding `answer`/`problem_type`/`question_type` metadata and removing the `synthetic_amc` subset after an ablation found it hurt performance [12]; its nine columns are `problem`, `solution`, `answer`, `problem_type`, `question_type`, `problem_is_valid`, `solution_is_valid`, `source`, `synthetic` - no `messages` column, so it does not carry this release's chat-ready TIR shape [13]. Not a TIR successor.
- No TIR-specific successor or cleaned re-release of this dataset was found in the sources checked.

## A row

One config and one message shape (`user` then `assistant`, both splits) [9][10], so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [9], with the Python execution trace truncated:

```json
{
  "problem": "What is the coefficient of $x^2y^6$ in the expansion of $\\left(\\frac{3}{5}x-\\frac{y}{2}\\right)^8$?  Express your answer as a common fraction.",
  "solution": "To determine the coefficient of \\(x^2y^6\\) in the expansion of \\(\\left(\\frac{3}{5}x - \\frac{y}{2}\\right)^8\\), we can use the binomial theorem. [...] ```python\nfrom math import comb\nbinom_coeff = comb(8, 6)\na_term = (3/5)**2\nb_term = (-1/2)**6\ncoefficient = binom_coeff * a_term * b_term\nprint(coefficient)\n```\n```output\n0.1575\n```\nThe coefficient of \\(x^2y^6\\) [...] is \\(\\frac{63}{400}\\). [...] \\[\n\\boxed{\\frac{63}{400}}\n\\]",
  "messages": [
    {"role": "user", "content": "What is the coefficient of $x^2y^6$ in the expansion of $\\left(\\frac{3}{5}x-\\frac{y}{2}\\right)^8$?  Express your answer as a common fraction."},
    {"role": "assistant", "content": "To determine the coefficient of \\(x^2y^6\\) [...] \\[\n\\boxed{\\frac{63}{400}}\n\\]"}
  ]
}
```

The `messages[1].content` field is the same text as `solution` in every sampled row [9][10].

## Where it came from

Built by the Numina project. NuminaMath-TIR is a direct extension of NuminaMath-CoT: the paper first built the 860k-problem CoT pool from Chinese K-12 exam papers, AoPS-forum and AMC/AIME problem statements, synthetic MATH/AMC-AIME-seeded problems, Orca-Math, and manually or regex-segmented olympiad-shortlist and national-contest PDFs, translating and reformatting each into a chain-of-thought solution with GPT-4o [2, Sec. 3.1]. To build TIR, the team extracted roughly 100K value-output problems from that CoT pool, sampled a GPT-4o-assistant-API solution per problem at temperature 0.8 in the ToRA tool-integrated-reasoning style, filtered out samples whose final answer did not match the reference, and repeated the process (three times, per the dataset card [1]) on the remaining problems - yielding this release's ~70k-problem, 72,540-row set [2, Sec. 3.3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the datasets-server endpoints used for neighbor row counts and this dataset's own size/info take no revision parameter and are live, not pinned.

[1] AI-MO/NuminaMath-TIR dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-TIR/raw/main/README.md - names GPT-4 as the generating model and states the negative-sample filtering repeated three times. Fetched 2026-08-11.

[2] Li et al., "NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions", 2024. https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf - the origin report; no arXiv record exists, so this is the paper as linked from the dataset card. Fetched 2026-08-11.

[3] AI-MO/NuminaMath-CoT dataset card (README), source-breakdown table. https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md Fetched 2026-08-11.

[4] Hugging Face Hub API record for AI-MO/NuminaMath-TIR. https://huggingface.co/api/datasets/AI-MO/NuminaMath-TIR?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=AI-MO%2FNuminaMath-TIR Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=AI-MO%2FNuminaMath-TIR Fetched 2026-08-11.

[7] Hugging Face Hub API, `downloadsAllTime` expansion. https://huggingface.co/api/datasets/AI-MO/NuminaMath-TIR?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] AI-MO/NuminaMath-7B-TIR model card (README). https://huggingface.co/AI-MO/NuminaMath-7B-TIR/raw/main/README.md - two-stage fine-tuning description, AIMO Progress Prize score. Fetched 2026-08-11.

[9] datasets-server first-rows endpoint, `train` split. https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2FNuminaMath-TIR&config=default&split=train Fetched 2026-08-11.

[10] datasets-server first-rows endpoint, `test` split. https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2FNuminaMath-TIR&config=default&split=test Fetched 2026-08-11.

[11] datasets-server first-rows endpoint for the neighbor AI-MO/NuminaMath-CoT, `train` split. https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2FNuminaMath-CoT&config=default&split=train Fetched 2026-08-11.

[12] AI-MO/NuminaMath-1.5 dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=AI-MO%2FNuminaMath-1.5 - live, unpinned. Fetched 2026-08-11.

[13] datasets-server info endpoint for AI-MO/NuminaMath-1.5. https://datasets-server.huggingface.co/info?dataset=AI-MO%2FNuminaMath-1.5 - column-name enumeration; live, unpinned. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with the AIME/AMC decontamination caution stated above carried forward before any scored run: the dataset's own construction draws on `amc_aime` and `aops_forum` competition pools whose decontamination against evaluation sets stopped at AIME 2024/AMC12 2023, and the served rows carry no source column to isolate that subset (see Quality). The screening row's flag names the same risk.

### The screening row

The row's own note [screening file, checked 2026-08-11]: "70k Numina human problems with tool-integrated (code) reasoning paths from GPT-4; `train` safe, `test` a small held-out sample." Its flag: "rule-risk: aime2025 - 70k problems selected from NuminaMath-CoT, whose source table lists amc_aime 4,072 and aops_forum 30,201 competition items."
