# GAIR/lima

A gated Hub repository of roughly 1,000 curated single-turn instruction/response pairs plus a held-out set of test prompts, the training data behind the LIMA fine-tuning experiments.

**GAIR/lima** hosts the training data for "LIMA: Less Is More for Alignment" [1], a paper from Meta AI, Carnegie Mellon University, the University of Southern California, and Tel Aviv University that fine-tuned a pretrained 65B LLaMA model on 1,000 hand-curated demonstrations, with no reinforcement learning or preference modeling, to test what the paper calls the Superficial Alignment Hypothesis [1]. It lives at https://huggingface.co/datasets/GAIR/lima . The paper reports the training set as 750 community-sourced examples (200 each from Stack Exchange STEM, Stack Exchange Other, and wikiHow, 150 from the Pushshift r/WritingPrompts subreddit, and 50 from Super-Natural Instructions) plus 250 prompt/response pairs the paper's authors wrote themselves, and a separate 300-prompt test set (70 from Pushshift r/AskReddit and 230 author-written) [1]. **The repository is gated ("auto" access) on the Hub, and this session's unauthenticated requests to its README, loading script, data files and Croissant/datasets-server endpoints all returned an access-restricted or 401/404 response, so this card cannot state the dataset's exact licence text, column schema, or served row counts from the files themselves - only what the Hugging Face repo-tree and parquet-config APIs and the paper expose without authentication.**

**Use it for**: SFT on single-turn instruction/response pairs (human-written or human-curated prompts with human-authored or human-edited answers), not preference pairs - map to the SFT method card. **Exact usage restrictions are unknown here because the licence text is gated and unread**; the Hub only exposes the licence tag `other` (custom, non-SPDX) [2].

**Licence**: tag `license:other` (custom, not a standard SPDX id); `cardData.license` is also `"other"` [2]. Repo access mode is `"gated": "auto"` [2]. The catch: this session could not read the actual licence terms - every unauthenticated attempt to fetch the README, the `lima.py` loading script, the Croissant metadata, or the resolved data files returned "Access to dataset GAIR/lima is restricted" (README/script/Croissant) or HTTP 401 (`resolve/main/train.jsonl`) [3][4][5].

**Shape**: the Hub repo-tree lists exactly two data files, `train.jsonl` (2,972,419 bytes) and `test.jsonl` (47,614 bytes), plus a `lima.py` loading script (2,081 bytes) and a 368-byte `README.md` [2]. The paper describes a design of 1,000 training examples (with a further 50-example held-out dev set drawn from the authors' own 250 prompts) and 300 test prompts [1]; because the repo is gated this session could not open either `.jsonl` file to confirm how many rows it actually contains or whether the 50-example dev set is folded into `train.jsonl` or omitted.

**Hold out**: `test.jsonl`, the paper's 300-prompt evaluation set (70 Pushshift r/AskReddit, 230 author-written) [1]; this session could not read the file to give a row count for what is actually served.

**Origin**: dataset curated by the authors of [1] (Meta AI, CMU, USC, Tel Aviv University); hosted on the Hub under the `GAIR` organization. Prompts are sourced from community forums (Stack Exchange, wikiHow, Reddit) and author-written prompts; responses are either mined from the highest-quality community answers or written by the paper's authors themselves - no model-generated content [1]. Hub API at the check date: `downloads` 7,532, `downloadsAllTime` 119,878, `likes` 469 [2][6].

**Trained-on-by**: not established by any source fetched for this card. The instruction-tuning survey "How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources" cites the LIMA paper only as related work in its references and does not list LIMA among the instruction datasets in its own training mixture (Table 1) [7].

**Introduced by**: [1] (Zhou, Liu, Xu, Iyer, Sun, Mao, Ma, Efrat, Yu, Yu, Zhang, Ghosh, Lewis, Zettlemoyer, Levy).

## Shape

The Hub repo-tree API (which is readable even though the repository is gated) lists five files [2]:

| file | bytes |
| --- | --- |
| `.gitattributes` | 2,265 |
| `README.md` | 368 |
| `lima.py` | 2,081 |
| `test.jsonl` | 47,614 |
| `train.jsonl` | 2,972,419 |

The Hub's `/parquet` endpoint confirms the loading script exposes one config, `plain_text`, with a `train` split and a `test` split, each backed by a single auto-converted parquet file - but the parquet files themselves return the same access-restricted error as the raw `.jsonl` files, so no column names, dtypes, or row counts could be read this session [8]. The Hub tag `size_categories:1K<n<10K` puts the total row count of the repository somewhere in that range [2], consistent with, but not proof of, the paper's stated ~1,000 train + 300 test design.

The paper states the total training data is roughly 750,000 tokens over exactly 1,000 sequences, and gives per-source average input/output lengths in characters: Stack Exchange (STEM) 200 examples, 117/523; Stack Exchange (Other) 200, 119/530; wikiHow 200, 12/1,811; Pushshift r/WritingPrompts 150, 34/274; Natural Instructions 50, 236/92; author-written (Group A) 200, 40/334; the 50-example dev set drawn from the same Group A pool has no output-length figure reported (`N/A`) [1]. No source read for this card states token or sequence-length statistics for the shipped `train.jsonl`/`test.jsonl` files as served.

## Quality

- Community answers (Stack Exchange, wikiHow) were filtered to exclude responses under 1,200 or over 4,096 characters, first-person phrasing ("I", "my"), or references to other answers ("as mentioned", "stack exchange"); links, images and other HTML were stripped, keeping only code blocks and lists [1].
- Reddit-sourced prompts (r/WritingPrompts, r/AskReddit) were manually curated rather than automatically filtered, because the paper's authors judge top-voted Reddit answers to often be "humorous or trolling" rather than reliably helpful [1].
- An ablation in the paper compares 7B models trained on filtered vs. unfiltered Stack Exchange data and finds "a significant 0.5 point difference" in favor of the filtered data, on the paper's human/ChatGPT-graded 1-6 quality scale [1].
- In the paper's human preference study, annotators found LIMA's responses equal to or better than those of GPT-4, Claude, and Bard in 43%, 46%, and 58% of compared cases respectively, and equal to or better than OpenAI's DaVinci003 (RLHF-trained) in 65% of cases; on an absolute scale, 88% of LIMA's responses were judged to meet the prompt's requirements and 50% were judged excellent [1]. GPT-4-as-annotator repeats corroborated these findings [1].
- The paper separately tests LIMA (fine-tuned with zero dialogue examples) on 10 live multi-turn conversations and finds it fails to follow the prompt within 3 turns in 6 of 10 conversations; adding 30 hand-crafted multi-turn dialogue chains (10 authored, 20 adapted from Stack Exchange comment threads) to form a 1,030-example set and refitting substantially improves multi-turn coherence [1]. That 30-dialogue supplement is not among the files this repository ships (only `train.jsonl` and `test.jsonl`) [2].
- No source read for this card states a measured contamination or duplicate rate for `train.jsonl`/`test.jsonl` as served; none is invented here.

## Load it

The repository requires authentication and an accepted access request (`"gated": "auto"`) [2]; `load_dataset` will fail without a Hub token that has been granted access:

```python
import datasets

REV = "68958e98267f5fb4a52a03ebcdae4ae59213fa7c"  # main at the check date
train = datasets.load_dataset("GAIR/lima", revision=REV, split="train", token=True)
test = datasets.load_dataset("GAIR/lima", revision=REV, split="test", token=True)
```

**Trap**: this session's unauthenticated `curl` calls to the README, `lima.py`, the Croissant endpoint, both `plain_text` parquet files, and `resolve/main/train.jsonl` every one returned an access-restricted error or HTTP 401 [3][4][5][8]; datasets-server's `/info`, `/size`, and `/first-rows` endpoints for this dataset all returned 404 for the same reason [9]. Without a token that has been granted access, none of these calls - including `load_dataset` itself - will succeed.

## Neighbors

- `64bits/lima_vicuna_format` reformats this dataset into Vicuna/ShareGPT-style `conversations` (a list of `{from, value}` turns) plus an `id` field, and is ungated; its card states it is "LIMA dataset in Vicuna ShareGPT format" derived from this repository, and its own datasets-server `/info` and `/size` report one `default` config, one `train` split, 1,030 rows, two columns (`conversations`, `id`) [10][11]. The 1,030-row count matches the paper's combined 1,000-example training set plus its 30-example multi-turn dialogue supplement [1], so this neighbor's `train` split appears to bundle both, not just the base 1,000-example set this card describes - a reader wanting only the single-turn 1,000 should not assume the two releases are row-for-row identical without checking further, and this session did not fetch a row from it to confirm.
- No other GAIR/lima re-release, cleaned variant, or successor was found among the sources checked for this card.

## A row

No row could be fetched live: every unauthenticated attempt to read `train.jsonl`, `test.jsonl`, the `plain_text` parquet files, or the datasets-server `/first-rows` endpoint for this dataset returned an access-restricted error, a 401, or a 404 (`{"error":"The dataset does not exist, or is not accessible without authentication (private or gated)..."}`) [3][4][8][9]. No fabricated row is given in its place.

For format illustration only - not a served row, and explicitly the paper's own figure rather than this card's data - the paper's Appendix A prints six training examples; one, sourced from Stack Exchange (STEM), begins: "What is the difference between minimum and infimum? I have a great confusion about this. The minimum is attained, the infimum isn't necessarily..." followed by a worked explanation [1].

## Where it came from

Built by the authors of [1], researchers then at Meta AI, Carnegie Mellon University, the University of Southern California, and Tel Aviv University, and uploaded to the Hub under the `GAIR` organization [1][2]. Community-forum prompts and answers were mined from Stack Exchange (STEM and non-STEM exchanges, quality- and diversity-sampled), wikiHow, and the Pushshift Reddit Dataset (r/WritingPrompts and r/AskReddit, manually curated); a further 50 prompts came from Super-Natural Instructions, lightly edited to fit the target response style; and 250 additional prompt/response pairs were written from scratch by the paper's authors, split between two author groups for training/dev and test respectively [1]. No response in the training or test data is model-generated; all content is either mined from human community answers or authored directly by the paper's authors [1].

## Sources

Every source below was fetched on the check date, 2026-08-11. The Hugging Face Hub repository for GAIR/lima is mutable and gated; the pinned `sha` in Load it is what its metadata endpoints reported at the check date, but this session could not authenticate to confirm the pin against the actual data files.

[1] Zhou, Liu, Xu, Iyer, Sun, Mao, Ma, Efrat, Yu, Yu, Zhang, Ghosh, Lewis, Zettlemoyer, Levy, "LIMA: Less Is More for Alignment", 2023. https://arxiv.org/abs/2305.11206 - the origin paper; read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2305.11206). Fetched 2026-08-11.

[2] Hugging Face Hub API record for GAIR/lima. https://huggingface.co/api/datasets/GAIR/lima?full=true - `sha`, `gated`, `cardData.license`, tags, `siblings` (repo-tree file list), `downloads`, `likes`, `lastModified`. Fetched 2026-08-11.

[3] GAIR/lima README (raw). https://huggingface.co/datasets/GAIR/lima/raw/main/README.md - returned "Access to dataset GAIR/lima is restricted. You must have access to it and be authenticated to access it." Fetched 2026-08-11.

[4] GAIR/lima loading script (raw). https://huggingface.co/datasets/GAIR/lima/raw/main/lima.py - same access-restricted response as [3]. GAIR/lima Croissant metadata: https://huggingface.co/api/datasets/GAIR/lima/croissant - same access-restricted response. Fetched 2026-08-11.

[5] GAIR/lima train.jsonl resolve attempt. https://huggingface.co/datasets/GAIR/lima/resolve/main/train.jsonl - HTTP 401. Fetched 2026-08-11.

[6] Hugging Face Hub API record for GAIR/lima with all-time downloads. https://huggingface.co/api/datasets/GAIR/lima?expand[]=downloadsAllTime - `downloadsAllTime`. Fetched 2026-08-11.

[7] Wang et al., "How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources", 2023. https://arxiv.org/abs/2306.04751 - read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2306.04751); cites the LIMA paper once, in its reference list, and does not include LIMA in its own Table 1 dataset mixture. Fetched 2026-08-11.

[8] Hugging Face Hub parquet-conversion index and files for GAIR/lima. https://huggingface.co/api/datasets/GAIR/lima/parquet - lists a `plain_text` config with one `train` and one `test` parquet file; both files themselves (`https://huggingface.co/api/datasets/GAIR/lima/parquet/plain_text/{train,test}/0.parquet`) returned the same access-restricted error as [3]. Fetched 2026-08-11.

[9] datasets-server info/size/first-rows endpoints for GAIR/lima. https://datasets-server.huggingface.co/info?dataset=GAIR%2Flima , https://datasets-server.huggingface.co/size?dataset=GAIR%2Flima , https://datasets-server.huggingface.co/first-rows?dataset=GAIR%2Flima&config=default&split=train - all three returned HTTP 404 with "The dataset does not exist, or is not accessible without authentication (private or gated)." Fetched 2026-08-11.

[10] 64bits/lima_vicuna_format dataset card (README) and Hub API record. https://huggingface.co/datasets/64bits/lima_vicuna_format/raw/main/README.md ; https://huggingface.co/api/datasets/64bits/lima_vicuna_format - ungated; states it reformats GAIR/lima into Vicuna ShareGPT format. Fetched 2026-08-11.

[11] datasets-server info/size endpoints for 64bits/lima_vicuna_format. https://datasets-server.huggingface.co/info?dataset=64bits%2Flima_vicuna_format ; https://datasets-server.huggingface.co/size?dataset=64bits%2Flima_vicuna_format - one `default` config, one `train` split, 1,030 rows, columns `conversations` (list of `from`/`value` string pairs) and `id`. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT data: the origin paper the card cites describes roughly 1,000 curated single-turn instruction/response pairs plus a held-out test set [1], matching the screening row's own description. The repository's Hub-side gating and the resulting inability to read its README, licence text, or served row counts (established throughout this card) are the reasons its exact licence terms and row counts remain unconfirmed, not a reason to distrust the row's classification.

### The screening row

The row's own note: "LIMA's roughly 1k curated single-turn instruction pairs - per the paper the card cites (arXiv 2305.11206), top-voted human content from Stack Exchange, wikiHow and Reddit plus author-written examples - shipping train.jsonl and a held-out test.jsonl of prompts; gated, licensed CC BY-NC-SA or stricter per source." This session could not independently verify the "licensed CC BY-NC-SA or stricter" clause of that note, since every attempt to read the actual licence text was blocked by the Hub's gate (see Licence above); the row carries no separate `flag`.
