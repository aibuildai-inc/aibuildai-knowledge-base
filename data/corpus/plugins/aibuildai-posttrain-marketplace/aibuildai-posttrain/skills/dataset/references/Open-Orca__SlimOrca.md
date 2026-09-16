# Open-Orca/SlimOrca

517,982 single-column ShareGPT-style conversations - GPT-4 completions on FLAN-Collection prompts - a curated, GPT-4-filtered subset of the larger Open-Orca/OpenOrca release.

**Open-Orca/SlimOrca** is a curated subset of Open-Orca's OpenOrca dataset, which augments prompts drawn from the FLAN Collection [2] with GPT-3.5/GPT-4 completions in the style described by Microsoft's Orca paper [1]. SlimOrca keeps only the ~500k GPT-4-completed entries from OpenOrca and adds a second GPT-4 pass that removes answers judged to disagree with the FLAN Collection's own human annotations, cutting the set to its current size while the dataset's own card claims comparable downstream quality to training on larger OpenOrca slices at two-thirds the compute [3]. Each row is one `conversations` list in system/human/gpt turn format, so it serves reasoning-trace SFT. It lives at https://huggingface.co/datasets/Open-Orca/SlimOrca .

**Use it for**: reasoning-trace SFT - the SFT method card - on the `conversations` column, a list of `{from, value, weight}` turns (system/human/gpt) in the ShareGPT chat dialect. No usage-shape restriction is stated by the card; no licence gate or non-commercial term applies [3][6].

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated (`"gated": false`, `"private": false`) [6]. No catch stated beyond the standard MIT grant.

**Shape**: 517,982 rows, one config (`default`), one split (`train`), one column (`conversations`) [7][8].

**Hold out**: nothing is stated as needing to be held out. Neither the dataset's own README [3], the parent OpenOrca README [4], nor the screening row's note [5] flags a contamination or eval-overlap risk for this release.

**Origin**: built and released by Open-Orca; completions are GPT-4 generations over FLAN Collection prompts, with a second GPT-4 pass acting as an automated filter against FLAN's human annotations [3]. Hub API at the check date: `downloads` 4,153 (live API), 4,658 recorded on the shortlist row; `likes` 300 [6].

**Trained-on-by**: the dataset's own README lists two demo fine-tunes on SlimOrca - `openaccess-ai-collective/jackalope-7b` and `Open-Orca/Mistral-7B-SlimOrca` [3]. Mistral-7B-SlimOrca's own model card confirms it fine-tuned Mistral 7B on SlimOrca and lists `Open-Orca/SlimOrca` in its `datasets` tag [9]. Jackalope-7b's model card says it used "the SlimOrca dataset, PIPPA, and various other open datasets" to fine-tune Mistral 7B, though its `datasets` tag names only `Open-Orca/OpenOrca` and other sources, not `Open-Orca/SlimOrca` [10].

**Introduced by**: no paper - the dataset card [3], which cites the Orca paper [1] and the FLAN Collection paper [2] as the methodology it reproduces.

## Shape

Rows and split (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 517,982 |
| total | 517,982 |

One config, `default`, with one column (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `conversations` | list\<struct\<from: string, value: string, weight: float64\>\> |

No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The dataset's own description states the defining quality step: an additional GPT-4 pass "to remove answers which appear wrong based on the human annotations from the FLAN dataset", which is what shrinks the set from OpenOrca's full size to ~500k [3]. No pass rate, agreement rate, or count of removed rows is stated.
- Reading 100 served rows at `train` offset 0 and 100 more at offset 517,882 (the end of the split) from the datasets-server rows endpoint [11]: in both samples every `human` turn carries `weight: 0.0`, every `gpt` turn carries `weight: 1.0`, and every `system` turn (present in 93/100 rows at offset 0, 100/100 at offset 517,882) carries `weight: null`. No source states what the `weight` field controls; the pattern above is what the 200 rows read actually show, not an inferred purpose.
- Turn count in the 100 rows read at offset 0 was 2 or 3 (a human/gpt pair, optionally preceded by a system turn); no source states this holds for the full 517,982 rows.
- No source states a measured duplicate rate for SlimOrca itself; a separate Open-Orca release, `SlimOrca-Dedup`, exists specifically to deduplicate it (see Neighbors), which implies duplicates were present, but no source quantifies the rate for this dataset's 517,982 rows.

## Load it

One split, one column, nothing to hold out. Pin the revision this card's numbers were read at (the shortlist's recorded commit, which matches the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-10-12) [6]:

```python
import datasets

REV = "ce9ed009ef3292bac33d3520afaf5ec804066120"  # main at the check date
train = datasets.load_dataset("Open-Orca/SlimOrca", revision=REV, split="train")  # 517,982 rows
```

**Trap**: the repository tree carries one served file, `oo-labeled_correct.gpt4.sharegpt.jsonl` [6], loaded as Parquet shards by `datasets-server`; the file name states the same "labeled correct" filtering step the README describes in prose, so the file itself is the source of the row count above, not a separate confirmation of it.

## Neighbors

All row counts below were read live from `datasets-server` at the check date [12].

- `Open-Orca/OpenOrca` - the full, unfiltered parent release this dataset is curated from, four columns (`id`, `system_prompt`, `question`, `response`) rather than one `conversations` column, and mixing GPT-3.5 and GPT-4 completions where SlimOrca keeps GPT-4 only [4][12]. Its row count is not a settled number: the `/size` response marks it `"partial": true` with `num_rows` 2,942,029 against a differing `estimated_num_rows` of 2,941,952, unlike the other three counts in this table which all return `"partial": false` [12]; the README itself says OpenOrca "currently represents a partial completion of the full intended dataset, with ongoing generation to expand its scope," and separately states its own current total as ~1M GPT-4 plus ~3.2M GPT-3.5 completions (~4.2M), above the 2,942,029 the live endpoint reports today; the README does not quantify the larger intended total this current figure itself falls short of [4]. Prefer SlimOrca over OpenOrca when GPT-4-only, chat-formatted, filtered data is what is wanted; use OpenOrca only for the larger, still-growing, unfiltered pool.
- `Open-Orca/SlimOrca-Dedup` - Open-Orca's own deduplicated build from SlimOrca: 363,491 rows, described by its card as removing RLHF instances and deduplicating "using minhash and Jaccard similarity techniques" [13]. Its card notes that the demo models trained on the full, non-deduplicated SlimOrca, not this variant [13].
- `Open-Orca/slimorca-deduped-cleaned-corrected` - a further-cleaned build on top of the deduplicated set: 181,745 rows, credited to an outside contributor and described by its card as stripping redundant prompt prefixes/suffixes (e.g. "Question:", "Answer:") from the deduplicated data [14].
- This corpus prefers the present, non-deduplicated `Open-Orca/SlimOrca` release, matching the shortlist row under review; choose a dedup neighbor only if duplicate rows are a concern for the target training run.

## A row

One config, one split, one schema. From `split="train"`, `row_idx=1` (datasets-server `/first-rows`) [11]:

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
      "weight": null
    },
    {
      "from": "human",
      "value": "Answer the following question: - number is 54    - debutteam is pittsburgh steelers    - draftpick is 166 [...] Given the details above, guess who could this information be about.\nAnswer:",
      "weight": 0.0
    },
    {
      "from": "gpt",
      "value": "The information provided seems to refer to Rian Wallace, a former NFL player.",
      "weight": 1.0
    }
  ]
}
```

## Where it came from

Built and released by Open-Orca. The parent OpenOrca dataset augments prompts drawn from the FLAN Collection [2] with completions from GPT-3.5 or GPT-4, following the distributions described in Microsoft's Orca paper [1]; OpenOrca's own card states it used the pre-generated FLAN Collection mirrors hosted under `conceptofmind` on the Hub rather than the original FLAN release, and that this left it with fewer entries than the Orca paper's stated totals for some submixes [4]. SlimOrca takes the subset of OpenOrca's entries completed by GPT-4 and applies a second GPT-4 pass that flags and removes answers disagreeing with the FLAN Collection's human annotations, producing the current 517,982-row set [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4", 2023. https://arxiv.org/abs/2306.02707 - the methodology SlimOrca and OpenOrca aim to reproduce; current title read from the live abs page. Fetched 2026-08-11.

[2] Longpre et al., "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning", 2023. https://arxiv.org/abs/2301.13688 - the prompt source both datasets augment; current title read from the live abs page. Fetched 2026-08-11.

[3] Open-Orca/SlimOrca dataset card (README). https://huggingface.co/datasets/Open-Orca/SlimOrca/raw/main/README.md - overview, GPT-4 filtering pass, demo models, citation. Fetched 2026-08-11.

[4] Open-Orca/OpenOrca dataset card (README). https://huggingface.co/datasets/Open-Orca/OpenOrca/raw/main/README.md - parent dataset's data fields, curation rationale, source-data caveats about FLAN submix coverage. Fetched 2026-08-11.

[5] The corpus screening row for `Open-Orca/SlimOrca`, supplied with this card's request - its `note`, read back in the appendix. Checked 2026-08-11.

[6] Hugging Face Hub API record for Open-Orca/SlimOrca. https://huggingface.co/api/datasets/Open-Orca/SlimOrca?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date, repository file tree (`siblings`). Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Open-Orca%2FSlimOrca Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Open-Orca%2FSlimOrca Fetched 2026-08-11.

[9] Open-Orca/Mistral-7B-SlimOrca model card (README). https://huggingface.co/Open-Orca/Mistral-7B-SlimOrca/raw/main/README.md - confirms SlimOrca fine-tune, `datasets` tag names `Open-Orca/SlimOrca`. Fetched 2026-08-11.

[10] openaccess-ai-collective/jackalope-7b model card (README). https://huggingface.co/openaccess-ai-collective/jackalope-7b/raw/main/README.md - states SlimOrca was used for fine-tuning; `datasets` tag does not list `Open-Orca/SlimOrca`. Fetched 2026-08-11.

[11] datasets-server first-rows and rows endpoints. https://datasets-server.huggingface.co/first-rows?dataset=Open-Orca%2FSlimOrca&config=default&split=train and https://datasets-server.huggingface.co/rows?dataset=Open-Orca%2FSlimOrca&config=default&split=train&offset=517882&length=100 - row samples at offset 0 and at the end of `train`. Fetched 2026-08-11.

[12] datasets-server size endpoint, one call per neighbor: `Open-Orca/OpenOrca`, `Open-Orca/SlimOrca-Dedup`, `Open-Orca/slimorca-deduped-cleaned-corrected`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[13] Open-Orca/SlimOrca-Dedup dataset card (README). https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup/raw/main/README.md - dedup method, RLHF-instance removal, demo-model caveat. Fetched 2026-08-11.

[14] Open-Orca/slimorca-deduped-cleaned-corrected dataset card (README). https://huggingface.co/datasets/Open-Orca/slimorca-deduped-cleaned-corrected/raw/main/README.md - describes prompt prefix/suffix cleaning applied on top of the deduplicated set. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with no holdout requirement stated by any source read: the dataset's own card describes it as GPT-4 completions on FLAN prompts, filtered by a second GPT-4 pass against FLAN's human annotations [3], matching the screening row's note [5], and no source flags eval-set overlap.

### The screening row

The row's own note [5]: "~518k GPT-4 completions on FLAN prompts, with a second GPT-4 pass dropping answers that disagree with FLAN's human annotations." The row carries no flag.
