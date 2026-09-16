# mlabonne/orpo-dpo-mix-40k

44,245 conversational preference pairs, each a shared prompt with a chosen and a rejected final assistant turn, assembled by blending seven existing DPO datasets into one ORPO/DPO training set.

**mlabonne/orpo-dpo-mix-40k** is a preference-pair dataset built by Maxime Labonne by combining `argilla/Capybara-Preferences` (7,424 samples), `argilla/distilabel-intel-orca-dpo-pairs` (2,299 samples, filtered to exclude GSM8K), `argilla/ultrafeedback-binarized-preferences-cleaned` (22,799 samples), `argilla/distilabel-math-preference-dpo` (2,181 samples), `unalignment/toxic-dpo-v0.2` (541 samples), `M4-ai/prm_dpo_pairs_cleaned` (7,958 samples), and `jondurbin/truthy-dpo-v0.1` (1,016 samples), with rule-based filtering removing 2,206 chosen answers that showed GPT-isms [1]. There is no origin paper; the dataset card is the only introducing source, alongside the author's blog post on ORPO fine-tuning Llama 3, which uses this dataset as its worked example [1][2]. **The mix bundles `unalignment/toxic-dpo-v0.2`, content written to elicit responses to illegal or harmful requests; the card gives a filter to drop it by the `source` column before any safety-sensitive run** [1]. It lives at https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k .

**Use it for**: preference-pair training for DPO or ORPO, with the toxic-dpo-v0.2 rows dropped first if the target run is safety-sensitive [1]. `chosen` and `rejected` are each a full message list (shared prompt turns plus one differing final assistant turn) — TRL's conversational, implicit-prompt preference format — while `prompt`/`question` duplicate the first user turn as a redundant string field, confirmed by inspecting served rows [3][4]. TRL's own trainer table recommends the explicit-prompt shape for its `ORPOTrainer` and `DPOTrainer`, so `extract_prompt()` should be applied before feeding this mix to either; see the DPO method card [4]. Maps to what the readme documents as an Axolotl `chat_template.argilla` type [1].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tags include `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [5]. The one catch: this licence is declared for the assembled mix itself; the card does not restate or reconcile the licences of the seven upstream datasets it draws from, so those upstream terms are not confirmed cleared here [1][5].

**Shape**: 44,245 rows, one config (`default`), one split (`train`) [6][7].

**Hold out**: nothing named. No source states a held-out eval split or a measured contamination rate for this mix; the one stated de-duplication step is that the orca subset was filtered to exclude GSM8K rows before inclusion [1].

**Origin**: assembled and released by Maxime Labonne (`mlabonne`) from seven third-party preference datasets [1]. How the individual chosen/rejected turns were originally generated or labeled by those upstream datasets is not stated by this card. Hub API at the check date: `downloads` 755, `likes` 308 [5].

**Trained-on-by**: `mlabonne/OrpoLlama-3-8B`, the author's own ORPO fine-tune of Llama-3-8B, whose model card lists `mlabonne/orpo-dpo-mix-40k` as its training dataset and names the same blog post as its origin [8]. Paging through the full Hub models search filtered to this dataset returns 247 repositories: 81 of their names contain "orpo" (OrpoLlama-3-8B and its quantizations, plus independent fine-tunes such as `dfurman/Llama-3-8B-Orpo-v0.1` and `Danielbrdz/Barcenas-14b-Phi-3-medium-ORPO`), and the remainder are a mix of other derivative families, including 45 repositories in the `mlabonne/NeuralDaredevil-8B-abliterated` / `Llama-3-8B-Instruct-abliterated-dpomix` family (22 of them `Zoyd/` exl2 quantizations, the other 23 further GGUF/AWQ/mlx builds and the two base uploads) [9].

**Introduced by**: no paper - the dataset card [1], plus the introducing blog post "Fine-tune Llama 3 with ORPO" [2].

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 44,245 |

One config, `default`, with five columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `source` | string |
| `chosen` | list\<struct\<content: string, role: string\>\> |
| `rejected` | list\<struct\<content: string, role: string\>\> |
| `prompt` | string |
| `question` | string |

Sizes (datasets-server `/size`) [6]: 126,503,374 bytes of original Parquet download, 442,777,103 bytes decoded in memory. No source states sequence-length or token statistics for this dataset.

Reading `train` at offset 0 (22 rows, first-rows endpoint) [3]: chosen/rejected message-list length ranges from 2 turns (single-turn Q&A, e.g. the `GOAT` and `TheoremQA` sources) to 10 turns (multi-turn dialogue, e.g. `Dove`) among the rows read; `prompt` and `question` were identical strings in every one of the 22 rows read, and each equaled the content of `chosen[0]` where the row was single-turn.

## Quality

- The `source` column carries 26 distinct labels across all 44,245 rows (datasets-server `/statistics`), a finer split than the seven datasets the card names: `prm_dpo_pairs` (7,985), `orca_dpo_pairs` (2,299), `distilabel-math-preference-dpo` (2,181), `toxic-dpo-v0.2` (541), and `truthy_dpo` (1,016) each match one of the card's five most-specific named sources one-to-one, while the remaining 21 labels (`sharegpt` 9,324, `ultrachat` 5,903, `evol_instruct` 4,113, `Airoboros` 1,630, `flan_v2_niv2` 1,917, `Dove` 1,888, `Know-Logic` 684, `TaskSource` 695, `GOAT` 593, `flan_v2_cot` 597, `EverythingLM` 510, `false_qa` 449, `GPT4LLM` 394, `flan_v2_p3` 395, `General-Instruct` 384, `Less-Wrong` 227, `SuperCOT` 196, `TheoremQA` 133, `flan_v2_flan2021` 101, `Verified-Camel` 81, `Tigerbot` 9) sum to exactly 30,223, matching the card's combined count for `argilla/Capybara-Preferences` (7,424) plus `argilla/ultrafeedback-binarized-preferences-cleaned` (22,799) — those two are themselves aggregations of the finer-grained upstream sub-sources [1][10]. The `prm_dpo_pairs` count of 7,985 in the served data does not exactly match the card's stated 7,958 for `M4-ai/prm_dpo_pairs_cleaned`, a 27-row difference; no source explains it [1][10].
- The card's only stated content caveat is the toxic-dpo-v0.2 warning quoted above, plus a code snippet showing how to filter it out by `source` [1].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for the assembled 44,245-row mix.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-10-17) [5]:

```python
import datasets

REV = "0f72511202b8f093e9be60e1683d84b046062e36"  # main at the check date
train = datasets.load_dataset("mlabonne/orpo-dpo-mix-40k", revision=REV, split="train")  # 44,245 rows
```

**Trap**: `chosen`/`rejected` are already implicit-prompt conversational lists, not the `{"prompt", "chosen", "rejected"}` explicit-prompt shape most preference trainers expect by default; run `trl.extract_prompt()` (or use the sibling `-flat` release below) before handing this to a trainer built around the explicit-prompt format [1][4].

## Neighbors

- `mlabonne/orpo-dpo-mix-40k-flat` — the same author's reformatting of the same 44,245 rows into plain strings: `chosen` and `rejected` hold only the final-turn text, `prompt`/`question` hold the shared prefix as a string, and a `system` column (all-null in the row read) is added. Its own card recommends it over this release specifically for DPO in Axolotl, saying it is easier to parse there, while recommending this release for ORPO [11].
- `arcee-ai/cleaned-mlabonne-orpo-dpo-mix-40k` — same row count (44,245) and the same list-of-dicts `chosen`/`rejected` shape, but drops the `prompt` column and adds `id` and an all-null `system` column; its README carries only front-matter metadata and states no cleaning method, so what changed beyond column reshaping is not stated [12][13].
- A live Hub search for the dataset's name returns 25 further derivatives at the check date (confirmed by raising the query's page size until no further results appeared) — language variants (`WKLI22/orpo-dpo-mix-zh-40k`, `yhavinga/orpo-dpo-mix-dutch-40k`, `arcee-globe/arabic-orpo-dpo-mix-40k-filtered`), a shortened cut (`Trelis/orpo-dpo-mix-40k-SHORT`), and format ports (`mlx-community/orpo-dpo-mix-40k-mlx`); none of these are described further here [14].

## A row

One config and split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], with the long chosen/rejected text truncated:

```json
{
  "source": "Airoboros",
  "prompt": "The setting is an otherworldly, yet eerily familiar, metropolis known as \"Zephyria.\" [...] Now, immerse me in this ethereal journey through Zephyria.",
  "question": "The setting is an otherworldly, yet eerily familiar, metropolis known as \"Zephyria.\" [...] Now, immerse me in this ethereal journey through Zephyria.",
  "chosen": [
    {"role": "user", "content": "The setting is an otherworldly, yet eerily familiar, metropolis known as \"Zephyria.\" [...]"},
    {"role": "assistant", "content": "As you step onto the teleportation platform, there's a momentary sense of disorientation [...]"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "rejected": [
    {"role": "user", "content": "The setting is an otherworldly, yet eerily familiar, metropolis known as \"Zephyria.\" [...]"},
    {"role": "assistant", "content": "As you step onto the teleportation platform, there's a momentary sense of disorientation [...]"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

`chosen` and `rejected` share every turn except the last: comparing the six turns of this row pairwise, the first five are identical between `chosen` and `rejected`, and only the sixth (final assistant) turn differs [3].

## Where it came from

Built and released on the Hub by Maxime Labonne (`mlabonne`). The card describes it as a combination of seven existing DPO datasets — `argilla/Capybara-Preferences`, `argilla/distilabel-intel-orca-dpo-pairs`, `argilla/ultrafeedback-binarized-preferences-cleaned`, `argilla/distilabel-math-preference-dpo`, `unalignment/toxic-dpo-v0.2`, `M4-ai/prm_dpo_pairs_cleaned`, and `jondurbin/truthy-dpo-v0.1` — crediting Argilla, Unalignment, M4-ai, and jondurbin for the source datasets, and states that rule-based filtering removed 2,206 chosen answers for showing GPT-isms [1]. The card documents no de-duplication or labeling work of its own beyond that filtering and the per-source score thresholds already applied by the upstream repositories, which for two of the seven kept only chosen answers scored 5 or higher [1]. The card's history section notes that earlier versions of the release are kept on separate branches (`v1.0`), which the blog post's worked example quotes different per-source sample counts from, consistent with describing an earlier branch rather than the current `main` blend read for this card [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] mlabonne/orpo-dpo-mix-40k dataset card (README). https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/raw/main/README.md — blend composition, filtering, licence, toxicity note, Axolotl usage, version history. Fetched 2026-08-12.

[2] Maxime Labonne, "Fine-tune Llama 3 with ORPO", Hugging Face blog. https://huggingface.co/blog/mlabonne/orpo-llama-3 — worked example using this dataset for an ORPO fine-tune of Llama 3. Fetched 2026-08-12.

[3] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mlabonne%2Forpo-dpo-mix-40k&config=default&split=train — 22 rows read at offset 0. Fetched 2026-08-12.

[4] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats — implicit- vs explicit-prompt preference format definitions, `extract_prompt()`, per-trainer recommended format table (`ORPOTrainer`/`DPOTrainer`: "Preference (explicit prompt recommended)"). A `main` build, unpinned and mutable. Fetched 2026-08-12.

[5] Hugging Face Hub API record for mlabonne/orpo-dpo-mix-40k. https://huggingface.co/api/datasets/mlabonne/orpo-dpo-mix-40k?full=true — licence, gate, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-12.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlabonne%2Forpo-dpo-mix-40k Fetched 2026-08-12.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mlabonne%2Forpo-dpo-mix-40k Fetched 2026-08-12.

[8] mlabonne/OrpoLlama-3-8B model card (README). https://huggingface.co/mlabonne/OrpoLlama-3-8B/raw/main/README.md — `datasets: - mlabonne/orpo-dpo-mix-40k` in front matter, stated as an ORPO fine-tune of Llama-3-8B created for the same blog post. Fetched 2026-08-12.

[9] Hugging Face Hub models API, filtered by dataset. https://huggingface.co/api/models?filter=dataset:mlabonne/orpo-dpo-mix-40k&limit=100 — live, unpinned list of model repositories tagged with this dataset; paginated through all three pages via the response's `Link: rel="next"` cursor to the end (247 repositories total). Fetched 2026-08-12.

[10] datasets-server statistics endpoint. https://datasets-server.huggingface.co/statistics?dataset=mlabonne%2Forpo-dpo-mix-40k&config=default&split=train — per-value frequency counts for the `source` column. Fetched 2026-08-12.

[11] mlabonne/orpo-dpo-mix-40k-flat dataset card (README) and datasets-server first-rows. https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k-flat/raw/main/README.md ; https://datasets-server.huggingface.co/first-rows?dataset=mlabonne%2Forpo-dpo-mix-40k-flat&config=default&split=train — flattened schema, DPO-vs-ORPO recommendation, one row read to confirm string vs list-of-dicts shape. Fetched 2026-08-12.

[12] arcee-ai/cleaned-mlabonne-orpo-dpo-mix-40k dataset card (README). https://huggingface.co/datasets/arcee-ai/cleaned-mlabonne-orpo-dpo-mix-40k/raw/main/README.md — front-matter-only card; schema read from its `dataset_info` block. Fetched 2026-08-12.

[13] datasets-server size endpoint for the same repository. https://datasets-server.huggingface.co/size?dataset=arcee-ai%2Fcleaned-mlabonne-orpo-dpo-mix-40k — row count. Fetched 2026-08-12.

[14] Hugging Face Hub datasets search API. https://huggingface.co/api/datasets?search=orpo-dpo-mix-40k&limit=100 — live, unpinned list of repositories whose name matches this dataset's; queried with a page size above the true result count (25 returned, no further pagination indicated), so the count is complete, not a partial page. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as a preference-pair (DPO/ORPO) training source, with the `toxic-dpo-v0.2` rows dropped first for a safety-sensitive run: the card's own filtering instructions and toxicity warning (quoted in the opening paragraph) establish that restriction [1], and no source names a held-out evaluation set this mix must avoid.

### The screening row

The row's own note [supplied with this card's request]: "blend of Argilla DPO sets (Capybara, distilabel-orca, UltraFeedback-cleaned, math, toxic-dpo)." The row carries no flag.
