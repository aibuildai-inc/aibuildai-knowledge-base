# openai/gsm8k

17,584 grade-school math word problems with human-written, multi-step natural-language solutions, split across a `main` configuration and a `socratic` variant that adds Socratic sub-questions to each solution step.

**openai/gsm8k** is OpenAI's GSM8K (Grade School Math 8K) release, introduced in "Training Verifiers to Solve Math Word Problems" [1] to diagnose and improve multi-step mathematical reasoning in language models. Each row pairs a `question` string with an `answer` string that walks through the arithmetic steps, annotated with calculator-style `<<...>>` markers and ending in a `#### <number>` final answer [2]. **The Hub repository is tagged `benchmark:official` with an eval-yaml configuration, marking `main/test` (1,319 rows) as this benchmark's official evaluation split [3]; treat it as held out and train only on `main/train` (7,473 rows). `socratic/train` and `socratic/test` mirror the same 7,473/1,319 split with Socratic-style sub-questions inserted into the `answer` field [2].** It lives at https://huggingface.co/datasets/openai/gsm8k .

**Use it for**: reasoning-trace SFT on `main/train` (or `socratic/train`) - each row's `answer` is a full worked solution ending in the numeric result, so it maps directly to a single-turn prompt/completion SFT format with `question` as the prompt and `answer` as the target completion. Never train on `main/test` or `socratic/test`; those are this benchmark's own evaluation rows [2][4]. See the SFT method card.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tags include `license:mit`, and the README states the dataset "is licensed under the MIT License") [2][3]. Ungated, public repo (`"gated": false`, `"private": false`) [3].

**Shape**: 17,584 rows across two configs (`main`, `socratic`), each with `train`/`test` splits of 7,473/1,319 rows, two string columns (`question`, `answer`) [5][6].

**Hold out**: `main/test` (1,319 rows) - the repository's `eval.yaml` names exactly one task, `config: main, split: test`, so this is the split the `benchmark:official`/`benchmark:eval-yaml` tags cover [3][7]. `socratic/test` (1,319 rows) is not named by `eval.yaml` or the Hub tags, but it mirrors `main/test`'s same underlying problems restated with sub-questions [2], so hold it out too rather than training on it. Train only on `main/train` and/or `socratic/train` (7,473 rows each).

**Origin**: built by OpenAI; questions and solutions are human-written, drafted by freelance contractors on Upwork and then scaled up with crowdworkers via Surge AI [2]. Hub API at the check date: `downloads` 946,606, `downloadsAllTime` 14,294,054, `likes` 1,556 [3].

**Trained-on-by**: the origin paper's own GPT-3 finetuning baseline, trained directly on the GSM8K training set as one of the two methods the paper compares against its proposed verifier [1]. MetaMath's MetaMathQA dataset is built entirely by rewriting and bootstrapping questions from the GSM8K and MATH training sets, with each item's `original_question` traceable back to the GSM8K (or MATH) train set, and is used to finetune the MetaMath-7B/70B and MetaMath-Mistral-7B models [8][9].

**Introduced by**: [1] (Cobbe et al.), plus the introducing blog the dataset card links as its homepage, https://openai.com/blog/grade-school-math/ [2].

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| config | split | rows |
| --- | --- | --- |
| `main` | `train` | 7,473 |
| `main` | `test` | 1,319 |
| `socratic` | `train` | 7,473 |
| `socratic` | `test` | 1,319 |
| total | | 17,584 |

Both configs share the same two-column schema (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `question` | string |
| `answer` | string |

Byte sizes (datasets-server `/size`) [5]: 5,889,887 bytes of original Parquet files total, 10,880,783 bytes decoded in memory; `main` alone is 2,725,633 bytes on disk / 4,709,831 in memory, `socratic` alone is 3,164,254 bytes on disk / 6,170,952 in memory. No source states sequence-length or token-count statistics for either config; the README states only that problems "take between 2 and 8 steps to solve" [2].

## Quality

- The paper's own re-annotation check (quoted in the dataset card from Appendix A) found that after workers re-solved every problem and disagreements were repaired or discarded, a second round of agreement checks on a subset still showed 1.7% of problems producing disagreements among contractors, which the authors estimate as the fraction of problems containing breaking errors or ambiguities, while noting a larger share may contain subtler errors [2].
- The `answer` field is annotated with `<<...>>` calculator markers around each arithmetic sub-step, a format the dataset card links to the source repository's own explanation and describes as chosen because natural-language solutions are "the most generally useful data format" for exposing a model's internal reasoning, quoting the paper [2].
- The dataset card states annotators were sourced through Surge AI (surgehq.ai) after an initial batch from Upwork freelancers, but marks the annotation process itself, curation rationale, and any PII review as "Needs More Information" [2].
- No source states a measured contamination or duplicate-row rate against other benchmarks; none is invented here. A downstream revision, GSM8K-Platinum, independently found enough test-set label and clarity issues to justify manually revising the full test set (see Neighbors) [10].

## Load it

Train on `main/train` and/or `socratic/train`; hold out both `test` splits; pin the revision this card's numbers were read at:

```python
import datasets

REV = "740312add88f781978c0658806c59bc2815b9866"  # main at the check date
train = datasets.load_dataset("openai/gsm8k", "main", revision=REV, split="train")          # 7,473 rows
test = datasets.load_dataset("openai/gsm8k", "main", revision=REV, split="test")             # 1,319 rows - hold out
socratic_train = datasets.load_dataset("openai/gsm8k", "socratic", revision=REV, split="train")  # 7,473 rows
```

**Trap**: `main` and `socratic` are separate configs covering the *same* 7,473 questions per split, restated with different `answer` formatting (plain step solutions vs. Socratic sub-questions) [2]. Loading both and concatenating their `train` splits for SFT duplicates every question's underlying problem, just with two different solution styles - decide up front whether that duplication is wanted rather than getting it by accident from loading both configs.

## Neighbors

- `madrylab/gsm8k-platinum` - a manually revised version of the `main/test` split only: frontier models were run on every example, mislabeled or ambiguous questions were fixed or removed, and the result is 1,209 rows (down from 1,319) in one `main/test` split, columns `question`, `answer`, `cleaning_status` [10][11][12]. It is an evaluation-set replacement, not a training corpus, and the dataset card says it is "a drop-in to replace the original gsm8k dataset" for eval [11].
- `meta-math/MetaMathQA` - a 395,000-row instruction dataset bootstrapped by rephrasing questions from the GSM8K and MATH training sets with an LLM, then filtering by answer-consistency; columns are `type`, `query`, `original_question`, `response` [8][13]. It overlaps this dataset's `main/train` at the level of underlying problems (each `original_question` traces back to a GSM8K or MATH train-set item) but is not row-identical to it, so mixing raw `main/train` SFT rows with MetaMathQA rows built from the same source problems risks near-duplicate training signal [8].
- `reasoning-machines/gsm-hard` - 1,319 rows built by replacing the numbers in GSM8K's questions with larger, less common ones, columns `input`, `code`, `target`; introduced alongside the PaL program-aided-language-model paper as a harder evaluation variant, not a training set [14][15][16].
- This corpus prefers `openai/gsm8k` itself for training (`main/train`, `socratic/train`) since it is the original, most-adopted release; reach for `madrylab/gsm8k-platinum` only if evaluating rather than training, since it revises `test`, not `train`.

## A row

Two distinct served shapes: `main` (both splits share one `answer` format) and `socratic` (both splits share the sub-question format). From `config="main"`, `split="train"`, `row_idx=0` [17]:

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

From `config="socratic"`, `split="train"`, `row_idx=0`, the same question, with the `answer` restructured into explicit sub-question/sub-answer pairs separated by `**` [18]:

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "How many clips did Natalia sell in May? ** Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nHow many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

`main/test`'s first served row confirms the same `answer` format as `main/train` (question, calculator-annotated steps, `####` final line), with a different problem [19]:

```json
{
  "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
}
```

## Where it came from

Built and released by OpenAI. The dataset card quotes the paper's Appendix A: an initial set of about a thousand problems and natural-language solutions was collected by hiring freelance contractors on Upwork, then the collection was scaled up with Surge AI, an NLP data-labeling platform; workers subsequently re-solved every problem (never their own), disagreements with the original solutions were repaired or discarded, and a further agreement-check round on a subset found 1.7% of problems still producing disagreements [2]. The `socratic` configuration restates the same problems with each solution step preceded by an explicit sub-question, per the dataset card's own field description [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021. https://arxiv.org/abs/2110.14168 - the origin paper; current title read from the live abs page and the paper's own description of the finetuning baseline trained on the GSM8K training set. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2110.14168). Fetched 2026-08-11.

[2] openai/gsm8k dataset card (README). https://huggingface.co/datasets/openai/gsm8k/raw/main/README.md - dataset summary, data instance formats for `main` and `socratic`, data fields, source-data quote from Appendix A, annotators, licence statement. Fetched 2026-08-11.

[3] Hugging Face Hub API record for openai/gsm8k. https://huggingface.co/api/datasets/openai/gsm8k?full=true - licence, gate, `sha`, `downloads`, `likes`, tags including `benchmark:official`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] The corpus screening row for `openai/gsm8k`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=openai%2Fgsm8k - this endpoint takes no revision parameter, so it reflects `main` live rather than the pinned `740312a` revision; the two agree as of the check date since the repo has not changed since `740312a` was written. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=openai%2Fgsm8k - same live-`main`, no-revision-parameter caveat as [5]. Fetched 2026-08-11.

[7] openai/gsm8k `eval.yaml`. https://huggingface.co/datasets/openai/gsm8k/raw/main/eval.yaml - the file behind the repository's `benchmark:eval-yaml` tag; it defines exactly one task, `config: main, split: test`, naming only `main/test` as the evaluation split. Fetched 2026-08-11.

[8] meta-math/MetaMathQA dataset card (README) and its origin paper, Yu et al., "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models", 2023. README: https://huggingface.co/datasets/meta-math/MetaMathQA/raw/main/README.md - states MetaMathQA data is augmented from the GSM8K and MATH training sets only, and that MetaMath-Mistral-7B is finetuned on it. Paper: https://arxiv.org/abs/2309.12284, read via ar5iv HTML (https://ar5iv.labs.arxiv.org/html/2309.12284) - question-bootstrapping method and reported GSM8K accuracy of MetaMath-7B (66.5%) and MetaMath-70B (82.3%). Fetched 2026-08-11.

[9] MetaMathQA README's "Model Details" section, same fetch as [8] - confirms MetaMath-Mistral-7B is "fully fine-tuned on the MetaMathQA datasets." Fetched 2026-08-11.

[10] Vendrow, Vendrow, Beery and Madry, "Do Large Language Model Benchmarks Test Reliability?", 2025. https://arxiv.org/abs/2502.03461 - current title read from the live abs page; this is the GSM8K-Platinum paper referenced by the neighbor's dataset card. Fetched 2026-08-11.

[11] madrylab/gsm8k-platinum dataset card (README). https://huggingface.co/datasets/madrylab/gsm8k-platinum/raw/main/README.md - revision method, row count, column names, "drop-in" framing. Fetched 2026-08-11.

[12] datasets-server size endpoint for madrylab/gsm8k-platinum. https://datasets-server.huggingface.co/size?dataset=madrylab%2Fgsm8k-platinum - row count (1,209); this endpoint takes no revision parameter, so the count is live, not pinned. Fetched 2026-08-11.

[13] datasets-server size and info endpoints for meta-math/MetaMathQA. https://datasets-server.huggingface.co/size?dataset=meta-math%2FMetaMathQA and https://datasets-server.huggingface.co/info?dataset=meta-math%2FMetaMathQA - row count (395,000) and column names (`type`, `query`, `original_question`, `response`); neither endpoint takes a revision parameter, so these are live, not pinned. Fetched 2026-08-11.

[14] reasoning-machines/gsm-hard dataset card (README). https://huggingface.co/datasets/reasoning-machines/gsm-hard/raw/main/README.md - construction method (replacing numbers with larger ones), column names, PaL paper link. Fetched 2026-08-11.

[15] datasets-server size endpoint for reasoning-machines/gsm-hard. https://datasets-server.huggingface.co/size?dataset=reasoning-machines%2Fgsm-hard - row count (1,319); this endpoint takes no revision parameter, so the count is live, not pinned. Fetched 2026-08-11.

[16] datasets-server info endpoint for reasoning-machines/gsm-hard. https://datasets-server.huggingface.co/info?dataset=reasoning-machines%2Fgsm-hard - column names and dtypes (`input`, `code`, `target`); no revision parameter, live not pinned. Fetched 2026-08-11.

[17] datasets-server first-rows endpoint, `main`/`train`. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fgsm8k&config=main&split=train Fetched 2026-08-11.

[18] datasets-server first-rows endpoint, `socratic`/`train`. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fgsm8k&config=socratic&split=train Fetched 2026-08-11.

[19] datasets-server first-rows endpoint, `main`/`test`. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fgsm8k&config=main&split=test Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as a reasoning-trace SFT source on `main/train` (and/or `socratic/train`) only. Two facts decide it, both established above: the repository's `eval.yaml` names `main/test` as this benchmark's own official evaluation split [3][7], and `socratic/test` should be held out alongside it since it mirrors the same underlying problems [2]; the screening row's note draws the same line between the trainable and forbidden splits [4].

### The screening row

The row's own note [4]: "The benchmark itself; `main/train` (7,473 human-written problems with step solutions) is exactly what the task permits, `main/test` (1,319) is forbidden, `socratic` mirrors both." The row carries no flag.
