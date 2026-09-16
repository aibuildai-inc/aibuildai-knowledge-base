# yizhongw/self_instruct

197,331 rows served across four configs - GPT-3-generated instruction/completion pairs, a small human-written evaluation set, and fixed subsamples of two existing instruction datasets - packaged as the official data release of the Self-Instruct paper.

**yizhongw/self_instruct** is the Hugging Face release of the training and evaluation data behind "Self-Instruct: Aligning Language Models with Self-Generated Instructions" [1], built by the paper's authors (Wang, Kordi, Mishra, Liu, Smith, Khashabi, and Hajishirzi). The paper's method starts from 175 human-written seed tasks and bootstraps new instructions, inputs, and outputs from GPT-3 ("davinci") itself, filtering the generations before using them for instruction tuning [1]; this repository packages that generated data (`self_instruct` config), the paper's own 252-task human evaluation set (`human_eval`), and fixed 50k-example subsamples of Super-NaturalInstructions [2] and P3 for comparison (`super_natural_instructions`, `p3`) [3]. **The `self_instruct` config is machine-generated and noisy: a manual review of 200 sampled instructions in the paper found only 54% had every field (instruction, input, output) valid, i.e. 46% had a problem in at least one field [1].** The dataset's own README repeats this 46% figure and advises users to be cautious with the data and consider filtering or improving it [3]. It lives at https://huggingface.co/datasets/yizhongw/self_instruct .

**Use it for**: instruction-following SFT (completion-style prompt/completion pairs, not a chat template) - maps to the SFT method card. Treat the noisy `self_instruct` config as unfiltered training data unless you add your own quality filter, and hold out `human_eval` and `super_natural_instructions`'s `test` split from training if you plan to evaluate with them.

**Licence**: apache-2.0 (`cardData.license`), ungated (`"gated": false`, `"private": false`) [4]. No further licence catch is stated by the card or the loading script.

**Shape**: 4 loadable configs, 197,331 rows total - `self_instruct/train` 82,612, `human_eval/train` 252, `p3/train` 52,657, `super_natural_instructions/train` 50,000 + `test` 11,810. The README's YAML also declares a fifth config, `prompt_source` (52,657 rows, byte-identical to `p3`), but the loading script defines no `BuilderConfig` for it, so it cannot be loaded [3][5] - see Shape below.

**Hold out**: `human_eval` (252 rows, the paper's own human-evaluation task set) and `super_natural_instructions/test` (11,810 rows, that config's declared test split) [3]. No source flags an overlap between `self_instruct`/`p3`/`super_natural_instructions/train` and any external eval benchmark.

**Origin**: built by the Self-Instruct paper's authors; `self_instruct` is GPT-3 (davinci) output, `human_eval` is human-written by the authors' team, and `p3`/`super_natural_instructions` are subsampled from existing crowd-sourced/expert-written datasets [1][3]. Hub API at the check date: `downloads` 859, `likes` 194 [4].

**Trained-on-by**: the paper's own GPT3_SELF-INST models are trained on this data - that is the dataset's origin experiment [1]. No source-confirmed adoption by a named third-party model or recipe was found beyond the origin paper.

**Introduced by**: [1] (Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions").

## Shape

Rows served, by config and split (datasets-server `/size`, `/info`) [5][6]:

| config | split | rows | columns |
| --- | --- | --- | --- |
| `self_instruct` | train | 82,612 | `prompt`, `completion` |
| `human_eval` | train | 252 | `id`, `motivation_app`, `instruction`, `instances` (list of `input`/`output`) |
| `p3` | train | 52,657 | `prompt`, `completion` |
| `super_natural_instructions` | train | 50,000 | `prompt`, `completion` |
| `super_natural_instructions` | test | 11,810 | `prompt`, `completion` |

Total served: 197,331 rows. The card's YAML `dataset_info` additionally declares a `prompt_source` config with 52,657 rows and the same byte sizes as `p3`; summing all five declared configs gives 249,988 rows, matching the card's declared total, but the dataset's loading script (`self_instruct.py`) lists only four `BUILDER_CONFIGS` (`self_instruct`, `human_eval`, `super_natural_instructions`, `p3`) and has no `_URLS` entry or generator for `prompt_source` [3][7]. It is dead metadata, not a fifth loadable shape; 249,988 minus the un-loadable 52,657 equals the 197,331 rows actually served [5][7].

Original download size 137,386,505 bytes; as Parquet 62,264,073 bytes; decoded in memory 128,114,471 bytes, across all four served configs (datasets-server `/size`) [5]. The paper's own statistics table (Table 1), for the generated data before this Hub packaging, reports 52,445 instructions and 82,439 instances with average lengths of 15.9 words (instruction), 12.7 words (non-empty input), and 18.9 words (output) [1]; the served `self_instruct` config's 82,612 rows is close to but not identical to the paper's reported 82,439 instances. No source states token-length statistics for the served rows.

## Quality

- The paper's manual quality review (200 sampled instructions, 1 instance each, reviewed by an expert annotator) found: instruction describes a valid task 92%, input appropriate for the instruction 79%, output correct and acceptable 58%, all three fields valid together 54% [1]. That 54%/46% pass/fail figure is the deciding number for the noise level of `self_instruct`.
- The dataset's README restates the same finding, that the paper's own manual review of 200 random instructions found 46% of the data points had a problem, and it recommends using the data with caution or filtering it further [3].
- The paper reports the resulting GPT3_Self-Inst model scores 39.9 ROUGE-L on unseen Super-NaturalInstructions tasks, versus 6.8 for vanilla GPT-3, 33.1 for T0 (11B, instruction-tuned without SuperNI), and 40.8 for InstructGPT-001 (davinci-001) - i.e. despite the noise, tuning on this data closes most of the gap between vanilla GPT-3 and InstructGPT-001 [1].
- No source states a duplicate-row rate or an annotator-agreement figure for `human_eval`, `p3`, or `super_natural_instructions` beyond the 46%-problem figure for `self_instruct`; none is invented here.

## Load it

Configs are separate `load_dataset` calls; pin the revision this card's numbers were read at (the Hub API's `sha`, matching the shortlisted commit; last modified 2023-03-07):

```python
import datasets

REV = "290f6a0851c1df3bd4057b21d988bab2c5c527e1"
self_instruct = datasets.load_dataset("yizhongw/self_instruct", "self_instruct", revision=REV, split="train")  # 82,612 rows
human_eval = datasets.load_dataset("yizhongw/self_instruct", "human_eval", revision=REV, split="train")        # 252 rows - hold out
p3 = datasets.load_dataset("yizhongw/self_instruct", "p3", revision=REV, split="train")                        # 52,657 rows
sni_train = datasets.load_dataset("yizhongw/self_instruct", "super_natural_instructions", revision=REV, split="train")  # 50,000 rows
sni_test = datasets.load_dataset("yizhongw/self_instruct", "super_natural_instructions", revision=REV, split="test")    # 11,810 rows - hold out
```

**Trap**: passing `data_dir` or `name="prompt_source"` fails, since the loading script defines no such config despite the README's YAML listing it - do not expect a fifth, P3-shaped shard [3][7]. The loading script also downloads each config's raw JSONL from the `main` branch of the linked GitHub repository at load time (`https://github.com/yizhongw/self-instruct`) rather than from files stored in this Hub repo, so a load can fail or silently drift if that upstream repository changes, independent of the Hub revision pin [7].

## Neighbors

- `bigscience/P3` - the full Public Pool of Prompts dataset that `p3` here is a fixed 52,657-row subsample of; the README names it directly [3].
- The Super-NaturalInstructions benchmark itself (not confirmed as a separate Hub dataset id here) - `super_natural_instructions` is a fixed 50k-train/11,810-test subsample of it, introduced by [2] [3].
- The upstream GitHub repository, https://github.com/yizhongw/self-instruct, is the authors' own release and the source the loading script fetches raw files from; it is the same data as this Hub repo, not an independent neighbor [3][7].
- No cleaned, binarized, or successor Hub release of this exact combined dataset was found.

## A row

Two distinct served shapes. `prompt`/`completion` pairs (used by `self_instruct`, `p3`, and both `super_natural_instructions` splits) and the structured `human_eval` shape.

From `config="self_instruct"`, `split="train"`, row 0 (datasets-server `/first-rows`) [8]:

```json
{
  "prompt": "Make a list of 10 ways to help students improve their study skills.\n\nOutput:",
  "completion": " 1. Make a schedule for studying and stick to it.\n2. Study in the same place every time.\n3. Set goals for yourself.\n4. Take breaks when you need them.\n5. Don't cram before an exam. [...] 10. Reward yourself after completing a task."
}
```

From `config="human_eval"`, `split="train"`, row 0 (datasets-server `/first-rows`) [9]:

```json
{
  "id": "user_oriented_task_0",
  "motivation_app": "Grammarly",
  "instruction": "The sentence you are given might be too wordy, complicated, or unclear. Rewrite the sentence and make your writing clearer by keeping it concise. Whenever possible, break complex sentences into multiple sentences and eliminate unnecessary words.",
  "instances": {
    "input": ["If you have any questions about my rate or if you find it necessary to increase or decrease the scope for this project, please let me know."],
    "output": ["If you have any questions about my rate or find it necessary to increase or decrease this project's scope, please let me know."]
  }
}
```

`p3/train` row 0 and both `super_natural_instructions` splits' row 0 were also read and confirmed to share the `self_instruct` shape above - `prompt` and `completion` string fields, no `instances` field [10][11][12].

## Where it came from

Built by the Self-Instruct paper's authors [1]. The `self_instruct` config comes from prompting GPT-3 ("davinci") with a bootstrapping pipeline that starts from 175 human-written seed tasks (one instruction and one instance each), generates new instructions, then generates instance inputs and outputs for them, filtering low-quality or overly similar generations before release [1]. The `human_eval` config is 252 tasks newly written by people motivated by real user-facing applications rather than existing NLP benchmarks, used for the paper's human evaluation [1][3]. The `p3` config is a fixed 52,657-example subsample of the Public Pool of Prompts (P3) dataset, and `super_natural_instructions` is a fixed 50,000-train/11,810-test subsample of the Super-NaturalInstructions dataset introduced in "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks" [2]; both are included so that Self-Instruct's results can be compared against these existing public datasets [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was fetched on the check date, 2026-08-12. Hub repositories are mutable, which is why Load it pins the revision (`sha`); the datasets-server endpoints used for row counts and sample rows take no revision parameter and reflect the same commit only because the Hub API's live `sha` for `main` matched the shortlisted commit at fetch time.

[1] Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions", 2022. https://arxiv.org/abs/2212.10560 - the origin paper; current title read from the live abs page; statistics (Table 1), quality review (Table 2), and Super-NI evaluation (Table 3) read from the arXiv HTML full text via ar5iv (https://ar5iv.labs.arxiv.org/html/2212.10560). Fetched 2026-08-12.

[2] Wang et al., "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks", 2022. https://arxiv.org/abs/2204.07705 - the source of the `super_natural_instructions` config, named in this dataset's README; current title read from the live abs page. Fetched 2026-08-12.

[3] yizhongw/self_instruct dataset card (README). https://huggingface.co/datasets/yizhongw/self_instruct/raw/main/README.md - config descriptions, YAML `dataset_info`, data instances, 46%-problem quality note, licensing/citation. Fetched 2026-08-12.

[4] Hugging Face Hub API record for yizhongw/self_instruct. https://huggingface.co/api/datasets/yizhongw/self_instruct?full=true - licence, gate/private status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=yizhongw%2Fself_instruct Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=yizhongw%2Fself_instruct Fetched 2026-08-12.

[7] Loading script for the dataset. https://huggingface.co/datasets/yizhongw/self_instruct/raw/main/self_instruct.py - `BUILDER_CONFIGS`, `_URLS`, per-config split generation; shows `prompt_source` has no config and that raw files are fetched from the GitHub repo at load time. Fetched 2026-08-12.

[8] datasets-server first-rows endpoint, `self_instruct` config. https://datasets-server.huggingface.co/first-rows?dataset=yizhongw%2Fself_instruct&config=self_instruct&split=train Fetched 2026-08-12.

[9] datasets-server first-rows endpoint, `human_eval` config. https://datasets-server.huggingface.co/first-rows?dataset=yizhongw%2Fself_instruct&config=human_eval&split=train Fetched 2026-08-12.

[10] datasets-server first-rows endpoint, `p3` config. https://datasets-server.huggingface.co/first-rows?dataset=yizhongw%2Fself_instruct&config=p3&split=train Fetched 2026-08-12.

[11] datasets-server first-rows endpoint, `super_natural_instructions` config, train split. https://datasets-server.huggingface.co/first-rows?dataset=yizhongw%2Fself_instruct&config=super_natural_instructions&split=train Fetched 2026-08-12.

[12] datasets-server first-rows endpoint, `super_natural_instructions` config, test split. https://datasets-server.huggingface.co/first-rows?dataset=yizhongw%2Fself_instruct&config=super_natural_instructions&split=test Fetched 2026-08-12.

[13] The corpus screening row for `yizhongw/self_instruct`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as SFT-shaped instruction data, with two conditions established above: the `self_instruct` config carries a documented ~46% per-field error rate and should be treated as unfiltered [1][3], and `human_eval` plus `super_natural_instructions/test` should be held out from training since they are the paper's own evaluation splits [3].

### The screening row

The row's note [13] explains that the `self_instruct` config is GPT-3 (davinci) output bootstrapped from 175 human seed tasks, that `human_eval` is 252 human-written user-oriented evaluation instructions, and that `super_natural_instructions` has its own test split. The row carries no flag.
