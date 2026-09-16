# stingning/ultrachat

A fully synthetic, large-scale corpus of multi-round dialogues in which two separate ChatGPT Turbo API instances play the user and the assistant, spanning three topical sectors, with roughly 1.5 million dialogues in the underlying release.

**stingning/ultrachat** is the Hub mirror of UltraChat, introduced by "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations" from Tsinghua's THUNLP/OpenBMB group [1]. The dataset card describes it as data generated so that "we do not directly use any data available on the Internet as prompts": one ChatGPT Turbo instance is prompted to act as the user and generate queries, the other plays the assistant and answers, across three sectors - open-ended world questions, writing/creation, and assistance on existing materials [2]. It serves multi-turn SFT-chat training: each row is a flat, alternating list of user/assistant turns starting with the user [2]. The repository itself was renamed on the Hub from `stingning/ultrachat` to `openbmb/UltraChat` at the same commit; the old slug still 307-redirects (its page resolves to HTTP 200 only once that redirect is followed) and its `datasets-server` info/size endpoints now return 404 with "The dataset has been renamed" - confirmed live at the check date - so all viewer-side numbers below were read live through the current name. The `datasets-server` endpoints used take no revision parameter, so those reads are live, not pinned; the Hub API separately confirms the repository's current `sha` matches this pinned commit at the check date, so the live reads happen to coincide with it right now [3]. **The Hub's automatic parquet conversion of this repository is itself partial: it materializes only 773,913 of the corpus's 1,468,348 dialogues (counted directly from the raw shard files at this commit, see Shape); a `load_dataset` call that lets the library fall back to that pre-converted parquet branch silently drops more than half the data, and the full corpus is only reachable by loading the raw `train_N.jsonl` shards.** It lives at https://huggingface.co/datasets/stingning/ultrachat .

**Use it for**: multi-turn SFT chat data - each row's `data` list is a bare sequence of alternating human/assistant turn strings with no role tags, so it must be reshaped into role-tagged messages (as HuggingFaceH4/ultrachat_200k does, turning it into `messages: [{role, content}, ...]`) before use; it is not a preference-pair format, there is no chosen/rejected column. See the SFT method card.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated [3]. No further restriction is stated beyond the MIT grant in the dataset card [2].

**Shape**: one config (`default`), one split (`train`); the datasets-server's own info/size endpoints report only 773,913 rows and mark the result `"partial": true`, but the ten raw `train_N.jsonl` shards served at this commit total 1,468,348 lines when counted directly [3][4][5].

**Hold out**: nothing - a single `train` split, and no source consulted (paper, dataset card, or screening note) flags an evaluation-set overlap.

**Origin**: built by the THUNLP/OpenBMB team; every turn is a ChatGPT Turbo API generation, with no human writing or labeling in the released dialogues [1][2]. Hub API at the check date (queried under the current name; downloads and likes are repo-level live counters, not tied to a specific commit): downloads 4,404, likes 500 [3].

**Trained-on-by**: the paper's own model, UltraLLaMA, an LLaMA-13B fine-tuned on this dataset with loss computed only on assistant turns [1]; openbmb/UltraLM-13b's model card lists `stingning/ultrachat` in its `datasets` field [6]; HuggingFaceH4/zephyr-7b-beta's model card states it was "initially fine-tuned on a filtered and preprocessed" version of the dataset it links directly to `stingning/ultrachat` [7], via the HuggingFaceH4/ultrachat_200k derivative [8].

**Introduced by**: [1] (Ding et al., "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations").

## Shape

Splits and rows, read two different ways. The `datasets-server` figures are live reads (the endpoint takes no revision parameter); the raw-shard figures are pinned to the commit `f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0` via the `x-repo-commit` response header on each shard request. The Hub API confirms this commit is also the current `sha` for `openbmb/UltraChat` at the check date, so the live and pinned reads coincide right now [3]:

| source | rows |
| --- | --- |
| datasets-server `/size` and `/info` (auto-converted parquet, marked `"partial": true`) | 773,913 |
| datasets-server's own `estimated_num_rows` extrapolation | 948,824 |
| raw `train_0.jsonl` .. `train_9.jsonl`, line-counted directly at this commit | **1,468,348** |

The raw-shard count is the true row count: it is a direct line count of the files this repository serves, not an estimate [4]. The parquet-auto-conversion undercount is visible in the Hub's own file naming - the repository's `refs/convert/parquet` revision stores its files as `default/partial-train/0000.parquet` through `0009.parquet`, explicitly labeled partial [9]. Consistent with this, the shortlisted scan's `size_categories` tag read `100K<n<1M` [12], while the live card's `cardData.size_categories` now reads `1M<n<10M` [3] - matching the 1,468,348 figure, not the smaller parquet-only count.

Two columns, both present in every row (datasets-server `/info`, and confirmed in the README's own example) [4][2]:

| column | dtype |
| --- | --- |
| `id` | string |
| `data` | list[string] |

Byte sizes: the ten raw `train_N.jsonl` shards total 9,288,990,895 bytes, summed from the `x-linked-size` header on each shard's resolve request at this commit [10]. The partial parquet conversion reports 2,513,422,942 bytes of parquet and 5,010,175,322 bytes decoded in memory for the 773,913 rows it processed [5]. No source states a token count for the released rows as a whole; the paper's own Table 8 statistical-analysis section states an average of 1,467.4 tokens per dialogue for UltraChat as it measured it in the paper, alongside the highest average turn count among the datasets it compared against [1].

## Quality

- The paper's construction pipeline includes an explicit filtering step: for the "Questions about the World" sector, roughly 500,000 questions were filtered and sampled as dialogue opening lines from a larger candidate pool; after generation, a further filtration step removes overly polite exchanges such as "Thank you," "Thanks," and "You're welcome" responses to make the dialogues read more like real user turns [1].
- The paper reports its own coherence and diversity evaluation: dialogues were scored 1-10 for coherence by ChatGPT, and UltraChat is reported to rank highest in coherence (tied with Baize) and highest in lexical diversity among the datasets it compared against, though it ranks slightly behind GPT4ALL on topic diversity [1].
- No source states a measured contamination rate, duplicate rate, or human-annotator-agreement figure for this dataset; every turn, both sides, is model-generated, so there is no human-label agreement to measure [1][2].
- Of the first 36 rows served at offset 0 of `train` (the only split), every row has an even-length `data` list (turn counts ranging 6 to 14), consistent with strict user/assistant alternation; this covers only those 36 rows at offset 0, not the full 1,468,348-row corpus [4].

## Load it

Pin the revision this card's numbers were read at, and load the raw shards explicitly rather than letting the library fall back to the Hub's partial parquet conversion:

```python
import datasets

REV = "f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0"  # main at the check date; same commit under both names
ds = datasets.load_dataset(
    "stingning/ultrachat",  # renamed on the Hub to openbmb/UltraChat; old slug still resolves
    data_files=[f"train_{i}.jsonl" for i in range(10)],
    revision=REV,
    split="train",
)  # 1,468,348 rows when the raw shards are loaded directly
```

**Trap**: this repository has no loading script, so a `load_dataset` call that omits `data_files` can be satisfied from the Hub's pre-converted `refs/convert/parquet` branch instead of the raw shards; that branch's files are named `default/partial-train/0000.parquet` .. `0009.parquet` and hold only 773,913 of the 1,468,348 dialogues [3][9]. Passing `data_files` as above forces the raw JSON shards and recovers the full corpus. A second trap: `datasets-server` calls made against the literal id `stingning/ultrachat` (the `/info` and `/size` endpoints) return HTTP 404 with "The dataset has been renamed" - re-issue any such calls against `openbmb/UltraChat` to reach the same, pinned-commit data [3].

## Neighbors

- `HuggingFaceH4/ultrachat_200k` - a heavily filtered subset reformatted for training: its own card states the original data comprises about 1.4 million ChatGPT-generated dialogues (close to, but not verified identical to, this release's 1,468,348) and that it truecased roughly 5% of turns and removed dialogues where the assistant claims to lack emotions or opinions. It ships as `prompt`/`prompt_id`/`messages` rows across four splits - `train_sft` 207,865, `test_sft` 23,110, `train_gen` 256,032, `test_gen` 28,304 (515,311 rows total) - and it, not this raw release, is what HuggingFaceH4/zephyr-7b-beta was fine-tuned on [8]. Prefer this release when the full unfiltered corpus or the raw three-sector split matters; prefer `ultrachat_200k` when a ready-to-train, role-tagged SFT/generation-ranking format is wanted directly.
- The original GitHub repository (`thunlp/UltraChat`) badges "Current Dialogues" at 1.57M and describes the same three sectors released incrementally (Questions about the World: 280k then 290k; Writing and Creation: an unstated first part then a further 457k) [11]; this figure is close to but not confirmed identical to this Hub mirror's pinned-commit count.

## A row

One config, one split, so one shape. From `config="default"`, `split="train"`, `row_idx=0`, read live through the datasets-server `/first-rows` endpoint under the current repository name (this endpoint takes no revision parameter; it coincides with the pinned commit because that is the repository's current `sha` at the check date) [4]:

```json
{
  "id": "0",
  "data": [
    "How can cross training benefit groups like runners, swimmers, or weightlifters?",
    "Cross training can benefit groups like runners, swimmers, or weightlifters in the following ways:\n\n1. Reduces the risk of injury: Cross training involves different types of exercises that work different muscle groups. [...]",
    "That makes sense. I've been wanting to improve my running time, but I never thought about incorporating strength training. Do you have any recommendations for specific exercises?",
    "Sure, here are some strength training exercises that can benefit runners: [...]",
    "Hmm, I'm not really a fan of weightlifting though. Can I incorporate other forms of exercise into my routine to improve my running time?",
    "Yes, absolutely! [...]"
  ]
}
```

`data` alternates user, assistant, user, assistant, ...; there are no separate role or turn-index fields, so a collator must reconstruct roles from list position.

## Where it came from

Built by the THUNLP/OpenBMB team and released on GitHub and the Hub together [1][2][11]. Every dialogue is machine-generated: two separate ChatGPT Turbo API instances are used, one prompted with instructions and personality cues to play the user and generate queries, the other to answer as the assistant, iterated to build up multi-turn context; the paper states this design explicitly avoids using any Internet text directly as a prompt, for privacy reasons [1][2]. The data is organized into three sectors - open-ended questions about real-world concepts and entities, writing/creation tasks from scratch, and assistance on existing materials such as rewriting, continuation, and summarization - released incrementally between March and April 2023 [1][2][11].

## Sources

Checked 2026-08-11; Hub repositories and `datasets-server` endpoints are mutable and carry no revision guarantee of their own, which is why Load it pins the commit `f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0` - the `sha` the Hub API returns for the repository (under either its old or current name) at the check date.

[1] Ding, Chen, Xu, Qin, Zheng, Hu, Liu, Sun, Zhou, "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations", 2023. https://arxiv.org/abs/2305.14233 - the origin paper; current title read from the live abs page; body read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2305.14233). Fetched 2026-08-11.

[2] openbmb/UltraChat dataset card (README), same file under either repository name; read from the `main` branch, which is live and unpinned but currently the same commit as the pin. https://huggingface.co/datasets/openbmb/UltraChat/raw/main/README.md - description, generation method, sector list, row-schema example, citation block. Fetched 2026-08-11.

[3] Hugging Face Hub API record for the repository, queried under both names. https://huggingface.co/api/datasets/openbmb/UltraChat?full=true and https://huggingface.co/api/datasets/stingning/ultrachat?full=true (the latter returns HTTP 307, redirecting to the former; a bare request to the repository page under the old slug, `https://huggingface.co/datasets/stingning/ultrachat`, likewise returns HTTP 307 and resolves to HTTP 200 only once that redirect is followed) - licence, gate status, `sha`, `cardData.size_categories`, downloads, likes, siblings, last-modified date. Fetched 2026-08-11.

[4] datasets-server `/info` and `/first-rows` endpoints, queried under the current name (the old name returns HTTP 404 with "The dataset has been renamed"). https://datasets-server.huggingface.co/info?dataset=openbmb%2FUltraChat and https://datasets-server.huggingface.co/first-rows?dataset=openbmb%2FUltraChat&config=default&split=train - these endpoints accept no revision parameter and reflect the repository's current default branch, live at the check date, not a pinned commit - features, `dataset_info.splits`, `"partial": true` flag, the 36-row sample used for the turn-count check, the sampled row. Fetched 2026-08-11.

[5] datasets-server `/size` endpoint, queried under the current name. https://datasets-server.huggingface.co/size?dataset=openbmb%2FUltraChat - this endpoint likewise accepts no revision parameter and is a live read of the current default branch - `num_rows` 773,913, `estimated_num_rows` 948,824, parquet and in-memory byte totals. Fetched 2026-08-11.

[6] Hugging Face Hub API record for the model openbmb/UltraLM-13b. https://huggingface.co/api/models/openbmb/UltraLM-13b - `cardData.datasets` lists `stingning/ultrachat`. Fetched 2026-08-11.

[7] HuggingFaceH4/zephyr-7b-beta model card (README). https://huggingface.co/HuggingFaceH4/zephyr-7b-beta/raw/main/README.md - states the model was fine-tuned on a filtered and preprocessed version of the dataset, linking to `stingning/ultrachat`. Fetched 2026-08-11.

[8] HuggingFaceH4/ultrachat_200k dataset card (README). https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/raw/main/README.md - states the original data comprises about 1.4M ChatGPT-generated dialogues, filtering description, per-split row counts, schema, and its use to train Zephyr-7B-beta. Fetched 2026-08-11.

[9] Hugging Face Hub API record for the repository's `refs/convert/parquet` revision. https://huggingface.co/api/datasets/openbmb/UltraChat/revision/refs%2Fconvert%2Fparquet - siblings named `default/partial-train/0000.parquet` through `0009.parquet`. Fetched 2026-08-11.

[10] Hub resolve endpoint, HEAD request per shard, at the pinned commit. https://huggingface.co/datasets/openbmb/UltraChat/resolve/main/train_{0..9}.jsonl - `x-linked-size` header per file (summed to 9,288,990,895 bytes) and `x-repo-commit: f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0` confirming the pin; the same ten files were then streamed and line-counted directly (`curl | wc -l`) to obtain the 1,468,348-row total in Shape. Fetched 2026-08-11.

[11] thunlp/UltraChat GitHub repository README. https://raw.githubusercontent.com/thunlp/UltraChat/main/README.md - "Current_Dialogues-1.57M" badge, incremental sector release dates and counts, GitHub-hosted download links. A `main`-branch file, unpinned and mutable. Fetched 2026-08-11.

[12] The corpus screening row for `stingning/ultrachat`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT chat data, with two caveats already established above: the raw `data` list carries no role tags and must be reshaped before training, and the corpus's true size (1,468,348 rows) is only reachable by loading the raw shards directly rather than the Hub's own partial parquet conversion (773,913 rows) [3][4][9][10]. Nothing in the paper, the dataset card, or the screening note flags evaluation-set overlap.

### The screening row

The row's own note [12]: "UltraChat, multi-round dialogues in which two ChatGPT Turbo APIs play the user and the assistant so no internet text is used directly as a prompt, in three sectors; the repo ships ten train_N.jsonl shards, so the viewer returns 404." The row carries no flag.
