# bigcode/commitpackft

702,062 real GitHub commits - old file, new file, and commit message, one file changed per commit - filtered down from a 4TB raw-commit pool to messages that read like natural-language instructions, spread across 277 programming-language configs.

**bigcode/commitpackft** ("CommitPackFT") is BigCode's filtered instruction-style commit dataset, introduced alongside its unfiltered parent CommitPack and the HumanEvalPack benchmark in "OctoPack: Instruction Tuning Code Large Language Models" [1]. Each row pairs a single-file commit's before/after code with its commit subject and message, giving an (instruction, input code, output code) triple that the paper uses to instruction-tune code models for editing tasks such as bug fixing [1]. The dataset card states no restriction beyond the described filtering, and the paper reports it explicitly checked the release against HumanEval and found no overlap [1]. It lives at https://huggingface.co/datasets/bigcode/commitpackft .

**Use it for**: SFT on code-editing instructions - the commit `subject` (or `message`) as the instruction, `old_contents` as the input code, `new_contents` as the target code, in the style the paper calls a "consistent schema to distinguish question and answer" for instruction tuning [1]. This is a row-per-example SFT format, not a preference pair or reward-model format; it maps to the SFT method card. Every one of the 277 configs shares the same 10-column schema (see A row below) [1][2].

**Licence**: repo-level SPDX tag `mit` [3]. Ungated, not private [3]. The catch: that repo-level MIT tag is not what governs each row - the dataset's own `license` column records the source repository's actual permissive licence per sample, one of 13 values the card lists (`mit`, `artistic-2.0`, `isc`, `cc0-1.0`, `epl-1.0`, `mpl-2.0`, `unlicense`, `unknown`, `apache-2.0`, `bsd-3-clause`, `agpl-3.0`, `lgpl-2.1`, `bsd-2-clause`) [2].

**Shape**: 702,062 rows, one `train` split per language, 277 language configs (no dedicated eval split anywhere in the repository) [2].

**Hold out**: nothing to hold out - every config ships only a `train` split, and the paper states it checked CommitPackFT for overlap with HumanEval solutions and docstrings and found none, attributing the clean result to the underlying commit data ending around 2016, years before HumanEval's 2021 release [1].

**Origin**: built and released by BigCode from GitHub commit metadata and code; the messages and code are human-authored (real commits), not model-generated [1]. Hub API at the check date: downloads 69,327, downloadsAllTime 3,877,856, likes 112 [3].

**Trained-on-by**: the origin paper's own OctoCoder (StarCoder 16B instruction-tuned on CommitPackFT + OASST) and OctoGeeX (CodeGeeX2 6B instruction-tuned on CommitPackFT + OASST); the paper's own ablation shows this pairing raises HumanEvalFix pass@1 from 23.1 (OASST alone) to 30.4 (CommitPackFT + OASST) [1][2]. No other adoption is stated in the sources checked here.

**Introduced by**: [1] (Muennighoff et al.).

## Shape

Rows and byte sizes as of the repository revision this card reads (`fc56fe33c030c6daa414c2b112c932b8eed085e6`) [3][4]:

| item | value |
| --- | --- |
| total rows | 702,062 |
| configs (languages) | 277 |
| split per config | `train` only |
| raw `data/<lang>/data.jsonl` bytes, summed across all 277 files | 1,580,402,782 |

That byte total is a direct sum of the LFS `data/*/data.jsonl` file sizes from the repository's file tree at this revision [4]; the dataset card's own Data Splits table separately reports a total of "1545.02" megabytes for the same 702,062-row total, a slightly different figure computed by the card's authors rather than read from the live tree [2]. The ten largest configs by row count, from that same table, are: `ruby` 69,413, `yaml` 114,320, `python` 56,025, `markdown` 62,518, `javascript` 52,989, `json` 39,777, `shell` 31,217, `text` 46,588, `php` 24,791, `java` 20,635 [2].

Columns (all string, confirmed by reading a live row - see A row): `commit`, `old_file`, `new_file`, `old_contents`, `new_contents`, `subject`, `message`, `lang`, `license`, `repos` [2][5].

The paper's own filter table bounds sequence length directly: samples are kept only where the concatenation of the code before, a special token, and the code after has between 50 and 768 tokens by the StarCoder tokenizer, and commit messages must be between 10 and 1,000 characters and more than 4 but fewer than 1,000 space-separated words [1]. A companion word-count table reports, before versus after this filtering: subject length 5.7 to 6.9 words, message length 8.7 to 9.9 words, and pre-commit code length 3,269.9 to 59.1 words on average [1].

## Quality

- The commit message is the human author's own message, not a generated label; the filters described below select for messages the paper judges to read like instructions, they do not add or verify a separate quality label [1].
- The paper reports its filters reduce CommitPack by "a factor of around 1000" to reach CommitPackFT's roughly 2GB / 277-language footprint from CommitPack's roughly 4TB / 350 languages [1].
- The paper explicitly checked for HumanEval contamination - no HumanEval solution or docstring was found present in CommitPackFT - and attributes this to the commit data's cutoff around 2016, well before HumanEval's release [1].
- The paper's own Table 8 gives, before versus after filtering, pre-commit code length 3269.9 to 59.1 words and post-commit code length 3269.8 to 77.6 words; it computes from these an after-filter code-change ratio of 77.6/59.1 = 1.31 (31%), and states this is larger than the before-filter ratio it reports as 3269.8/3269.9 = 1.007 (0.7%) - the second ratio as printed in the paper does not correspond to computing 3269.8/3269.9, which independently comes to about 0.99997 (an essentially unchanged, not increased, value); the discrepancy sits in the paper's own text and is not resolved here [1].
- On a Python subset the paper describes as roughly 56K-59K samples (the figure caption says "59K samples," the body text says "56K samples," and this release's `python` config has 56,025 rows), GPT-4 classification finds around 20% of commits are bug fixes [1]. The paper's 1-shot classification prompt allows each commit to fall into one or more of 18 categories - bug fixes, new features, refactoring/code cleanup, documentation, testing, user interface, dependencies, configuration, build system/tooling, performance improvements, formatting/linting, security, technical debt repayment, release management, accessibility, deprecation, logging/instrumentation, and internationalization - but the paper states a measured percentage only for the bug-fixes category [1].
- The paper's own instruction-data ablation on the Python split of HumanEvalPack shows the deciding number for this dataset's contribution: instruction-tuning StarCoder on OASST alone scores 23.1 pass@1 on HumanEvalFix, while adding CommitPackFT (CommitPackFT + OASST) raises that to 30.4, a 7.3-point gap on the code-repair task, alongside 35.1 vs. 34.5 on HumanEvalExplain and 46.2 vs. 46.4 on HumanEvalSynthesize (37.2 vs. 34.7 average) [1].
- No source states a duplicate-row rate or an inter-annotator agreement figure, because there is no separate annotation step beyond the automated filters described above; none is invented here.

## Load it

```python
import datasets

REV = "fc56fe33c030c6daa414c2b112c932b8eed085e6"  # main at the check date
python_train = datasets.load_dataset("bigcode/commitpackft", "python", revision=REV, split="train")  # 56,025 rows
```

**Trap**: this repository has 277 separate configs, one per language (`python`, `javascript`, `c#`, `c++`, ...) - `load_dataset("bigcode/commitpackft")` with no config name will fail or only work through the auxiliary loading script; you must pass a specific language config name (matching the `data/<lang>/` directory), and each config exposes only a `train` split, no `test` or `validation` [2][4].

## Neighbors

- `bigcode/commitpack` - the unfiltered 4TB parent, "a 4TB dataset of commits scraped from GitHub repositories that are permissively licensed," covering 350 languages before CommitPackFT's language- and quality-filtering down to 277 [1][6]. Use this only for large-scale pretraining, as the paper itself does for a SantaCoderPack ablation, not for instruction SFT [1].
- `bigcode/commitpackmeta` - GitHub metadata for the CommitPack commits (its own card description states it is "GitHub metadata for" CommitPack), not the code content itself [7].
- `bigcode/commitpack-subset-cf` - a CommitPack subset the paper's SantaCoderPack pretraining ablation used, restricted to six languages and to commits whose before+special-token+after fits in 8,192 tokens, formatted as `<commit_before>code_before<commit_message>commit_message<commit_after>code_after` [1][8].
- `rombodawg/Rombodawgs_commitpackft_Evolinstruct_Converted` - a third-party reformatting of this release into an Evol-Instruct-style schema; its own card states it is sourced directly from `bigcode/commitpackft` [9]. This corpus prefers the original bigcode/commitpackft release for reproducibility of the row counts and schema documented above.

## A row

One config/split shape covers the whole repository (277 configs, each a single `train` split, sharing the same 10 columns). From `config="python"`, `split="train"`, row 0, fetched from the served `data/python/data.jsonl` file at the pinned revision `fc56fe33c030c6daa414c2b112c932b8eed085e6`, with long fields truncated by an ellipsis marker [5]:

```json
{
  "commit": "e905334869af72025592de586b81650cb3468b8a",
  "old_file": "sentry/queue/client.py",
  "new_file": "sentry/queue/client.py",
  "old_contents": "\"\"\"\nsentry.queue.client\n~~~~~~~~~~~~~~~~~~~\n\n:copyright: (c) 2010 by the Sentry Team, see AUTHORS for more details.\n:license: BSD, see LICENSE for more details.\n\"\"\"\nfrom kombu import BrokerConnection\n...",
  "new_contents": "\"\"\"\nsentry.queue.client\n~~~~~~~~~~~~~~~~~~~\n\n:copyright: (c) 2010 by the Sentry Team, see AUTHORS for more details.\n:license: BSD, see LICENSE for more details.\n\"\"\"\nfrom kombu import BrokerConnection\n...",
  "subject": "Declare queues when broker is instantiated",
  "message": "Declare queues when broker is instantiated\n",
  "lang": "Python",
  "license": "bsd-3-clause",
  "repos": "imankulov/sentry,BuildingLink/sentry,zenefits/sentry,korealerts1/sentry,kevinastone/sentry,fotinakis/sentry,fuziontech/sentry,ngonzalvez/sentry,mvaled/sentry,Kronuz/django-sentry,..."
}
```

`repos` is a comma-separated list because the same commit can appear in multiple GitHub mirrors/forks; this one row lists 2,780 characters of repository names.

## Where it came from

BigCode built CommitPack from commit metadata in the GitHub Archive dump on Google BigQuery, applying quality filters, filtering for commercially friendly licenses, and discarding commits touching more than one file so each row's commit message stays specific to a single file's change; the filtered metadata was then used to scrape the affected file's contents before and after the commit from GitHub, producing the roughly 4TB, 350-language CommitPack [1]. To build CommitPackFT, BigCode applied additional strict filters on top of CommitPack - length, difference, extension, filename, message-length, word-count, cleaning, capitalization, token-count (50-768 StarCoder tokens), a start-word allowlist, and a noise-phrase blocklist - detailed in the paper's Appendix D, reducing the data by roughly a factor of 1,000 to about 2GB across 277 languages [1].

## Sources

Every source below was fetched on the check date, 2026-08-12; that date covers every number, quote, and row above. The Hub repository is mutable (it can be force-pushed), which is why Load it pins the revision `fc56fe33c030c6daa414c2b112c932b8eed085e6`.

[1] Muennighoff et al., "OctoPack: Instruction Tuning Code Large Language Models," 2023. https://arxiv.org/abs/2308.07124 - the origin paper introducing CommitPack, CommitPackFT, and HumanEvalPack; current title read from the live abs page. Body text and appendices read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2308.07124 . Fetched 2026-08-12.

[2] bigcode/commitpackft dataset card (README). https://huggingface.co/datasets/bigcode/commitpackft/raw/main/README.md - dataset summary, data fields, per-language Data Splits table, licensing note. Fetched 2026-08-12.

[3] Hugging Face Hub API record for bigcode/commitpackft. https://huggingface.co/api/datasets/bigcode/commitpackft?full=true and the same endpoint expanded with `downloadsAllTime` - sha, license tag, gated status, downloads, likes, last-modified date. Fetched 2026-08-12.

[4] Hugging Face Hub API recursive file tree for bigcode/commitpackft. https://huggingface.co/api/datasets/bigcode/commitpackft/tree/main?recursive=true - per-file byte sizes, summed for the total raw JSONL byte count above. Fetched 2026-08-12.

[5] `data/python/data.jsonl` served directly from the repository at the pinned revision. https://huggingface.co/datasets/bigcode/commitpackft/resolve/fc56fe33c030c6daa414c2b112c932b8eed085e6/data/python/data.jsonl - read as a partial byte-range fetch of the first rows; row 0 quoted above. A separate fetch of the same path via the live `main` ref returned byte-identical content at the check date. Fetched 2026-08-12.

[6] Hugging Face Hub API record for bigcode/commitpack. https://huggingface.co/api/datasets/bigcode/commitpack?full=true - card description. Fetched 2026-08-12.

[7] Hugging Face Hub API record for bigcode/commitpackmeta. https://huggingface.co/api/datasets/bigcode/commitpackmeta?full=true - card description. Fetched 2026-08-12.

[8] Hugging Face Hub API record for bigcode/commitpack-subset-cf. https://huggingface.co/api/datasets/bigcode/commitpack-subset-cf?full=true - card description of the SantaCoderPack pretraining subset and its commit-format template. Fetched 2026-08-12.

[9] Hugging Face Hub API record for rombodawg/Rombodawgs_commitpackft_Evolinstruct_Converted. https://huggingface.co/api/datasets/rombodawg/Rombodawgs_commitpackft_Evolinstruct_Converted?full=true - card description naming bigcode/commitpackft as its source. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as SFT data for code-editing instruction tuning. Two facts already established above decide it: the dataset's own commit-derived rows give a clean (instruction, input code, output code) triple with no eval split to accidentally train on, and the origin paper's explicit HumanEval overlap check found nothing [1].

### The screening row

The row's own note [screening index, bigcode/commitpackft, checked 2026-08-12]: "700k real GitHub commits (message plus before/after code), filtered to commit messages that read like instructions." The row carries no flag.
