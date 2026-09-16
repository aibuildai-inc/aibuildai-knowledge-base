# hkust-nlp/deita-10k-v0

10,000 multi-turn assistant conversations, automatically selected from a larger candidate pool for SFT alignment training, released as the Deita project's flagship 10K subset.

**hkust-nlp/deita-10k-v0** is built by the HKUST NLP group and released alongside "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning" [1], which introduces Deita ("Data-Efficient Instruction Tuning for Alignment"): a pipeline that scores candidate instruction-response pairs on complexity, quality, and diversity and selects a small high-scoring subset for supervised fine-tuning [1]. The dataset card states the 10k rows are "mainly automatically selected" from three source pools - a 58K ShareGPT pool, a 105K UltraChat sample, and a 143K WizardLM evolved-instruction pool [2]. **The rows served in the `train` split, read in full (all 10,000), come from only two of those three named pools: 8,724 rows tagged `source="ShareGPT"` and 1,276 tagged `source="UltraChat"`; no row in the served split carries a WizardLM source tag, though the card's prose lists WizardLM as one of the three pools selection drew from [2][8].** It lives at https://huggingface.co/datasets/hkust-nlp/deita-10k-v0 .

**Use it for**: multi-turn SFT chat training - the rows are single-column conversation transcripts (human/gpt turn lists), the `conversations` dialect; map to the SFT method card's chat format directly, no `extract_prompt` or pairing step needed. No usage-shape restriction is stated beyond the source-pool caveat above.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`), ungated [2][3]. The one catch: MIT is hkust-nlp's licence on its own selection/release, not a re-grant of the underlying conversations' licences - the card itself flags one of the three named source pools, ShareGPT, as "Apache 2.0 listed, no official repo found" [2], i.e. unverified upstream terms.

**Shape**: 10,000 rows, one config (`default`), one split (`train`); columns `id` (int64), `conversations` (list of `{from, value}`), `source` (string) [4][5].

**Hold out**: nothing. There is only a `train` split, no shipped eval/test split, and no source here or in the paper flags overlap between these rows and an evaluation benchmark [1][2][4].

**Origin**: built by hkust-nlp via the Deita automatic-selection pipeline over existing conversation pools; the conversation content itself is not hkust-nlp-authored but drawn from ShareGPT and UltraChat, whose turns were produced by a mix of human users and assistant models (ShareGPT: real users and ChatGPT; UltraChat: model-generated dialogues) per the card's pool descriptions [2]. Hub API at the check date: `downloads` 337, `downloadsAllTime` 6,903, `likes` 30 [3][6].

**Trained-on-by**: the origin paper's own DEITA-LLaMA1-13B-v1.0-sft, DEITA-LLaMA2-13B-v1.0-sft, and DEITA-7B-v1.0-sft (10K) models, per the dataset card's own performance table, which reports these models trained on "10K SFT" scoring 6.60/6.79/7.32 MT-Bench respectively against a random-10K baseline of 6.03/5.78/5.89 [2]. No adoption by models outside the Deita project itself is documented in the fetched sources.

**Introduced by**: [1] (Liu et al.).

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 10,000 |

One config, `default`, three columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `conversations` | list\<struct\<from: string, value: string\>\> |
| `source` | string |

Sizes (datasets-server `/size`) [4]: 387,833,875 bytes of original JSON download, 143,623,997 bytes as Parquet, 360,657,428 bytes decoded in memory. No source states token-length statistics. Reading the full served `train` split (all 10,000 rows) directly from the Hub's converted Parquet file [8]: message-count per conversation (each `{from, value}` entry counted as one message, not one exchange) ranges from 8 to 972, mean 44.0, median 30; `id` runs 0-9999 with no duplicates. Message counts differ sharply by source: ShareGPT rows average 48.5 messages, UltraChat rows average 13.6 [8].

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset.
- The only quality signal in the fetched sources is the downstream performance table on the dataset card: DEITA-7B-v1.0-sft (10K) scores 7.32 MT-Bench and 81.67% AlpacaEval versus a random-10K-sample baseline of 5.89 MT-Bench and 56.90% AlpacaEval on the same Mistral-7B base, which the card presents as evidence the automatic selection outperforms random sampling at equal data size [2].
- The card gives no per-row quality score, complexity score, or diversity score alongside the released rows themselves; those scorer outputs live in the separate `hkust-nlp/deita-quality-scorer-data` and `hkust-nlp/deita-complexity-scorer-data` releases, not in this one [7].

## Load it

Single split, no held-out set to separate; pin the revision this card's numbers were read at (the Hub API's `sha` for `main`, matching the shortlisted commit; the repo was last modified 2023-12-31) [3]:

```python
import datasets

REV = "fdb88575d6d3b0e5b8fc9934778b62313c7837ba"  # main at the check date
train = datasets.load_dataset("hkust-nlp/deita-10k-v0", revision=REV, split="train")  # 10,000 rows
```

**Trap**: the dataset card's prose says selection drew from three pools including WizardLM, but the actual served `source` column, read across all 10,000 rows, contains only `"ShareGPT"` and `"UltraChat"` values - a `groupby("source")` or filter expecting a WizardLM subset will silently return nothing [2][8].

## Neighbors

- `hkust-nlp/deita-6k-v0` - the same builder's 6,000-row sibling, one `train` split, with the same three columns `id`/`conversations`/`source` per its own `/info` record; this is the smaller Deita release the paper's 6K-SFT models train on, and is the natural pick when a smaller SFT set is wanted [9].
- `hkust-nlp/deita-quality-scorer-data` (489 rows, one `validation` split, columns `input`/`target`) and `hkust-nlp/deita-complexity-scorer-data` (499 rows, one `validation` split, columns `input`/`target`) - training/eval data for the Deita scoring models themselves, not conversation data for SFT; different shape and use than this release [7].
- `hkust-nlp/deita-redundant-pool-data` (100,000 rows, one `train` split, columns `id`/`conversations`) - the larger unselected candidate pool the Deita selection algorithm draws from and prunes for redundancy; not itself curated for training, so prefer this 10K release over the pool for SFT [7].
- This dataset itself is the corpus's preferred pick when 10K rows are wanted; reach for `deita-6k-v0` only when a smaller set is needed, since mixing the two would duplicate rows (both are drawn from overlapping source pools by the same selection method) [2].

## A row

One config, one split, one schema. From `split="train"`, `row_idx=0` of the Hub's converted Parquet file, read directly (not the datasets-server preview, which truncates this field) [8]:

```json
{
  "id": 0,
  "conversations": [
    {"from": "human", "value": "what is a pokemon type game called basicially? pokemon type game being where you find creatues, grow them up, make them breed each other, evolution, fight each other, etc"},
    {"from": "gpt", "value": "The type of game you are describing is commonly referred to as a \"monster collecting\" or \"creature collecting\" game. However, since the concept of these games was popularized by the Pokemon series, many people simply refer to them as \"Pokemon-like\" or \"Pokemon-inspired\" games. Some other examples of games in this genre include Digimon, Yo-Kai Watch, and Monster Rancher."}
  ],
  "source": "ShareGPT"
}
```

This row has 2 messages; the full split ranges from 8 to 972 messages per row, so most served rows are considerably longer multi-turn transcripts than this example [8].

## Where it came from

Built by hkust-nlp as part of the Deita project. The paper frames the release as the output of an automatic selection pipeline: candidate instruction-response samples are scored along three axes - complexity, quality, and diversity - using an Evol-based scoring approach, and a small high-scoring subset is chosen for SFT [1]. The dataset card names the three candidate pools the selection drew from - a 58K-conversation ShareGPT pool, a 105K-sample draw from UltraChat, and a 143K-sample pool of WizardLM's evolved Alpaca/ShareGPT instructions - though the served rows in this release carry only ShareGPT and UltraChat source tags, as established above [2][8]. The paper also releases the redundant candidate pool and the scorer training data as separate Hub datasets [7].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Liu, Zeng, He, Jiang, He, "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning", 2023. https://arxiv.org/abs/2312.15685 - the origin paper; current title read from the live abs page and the paper's HTML full text. Fetched 2026-08-11.

[2] hkust-nlp/deita-10k-v0 dataset card (README). https://huggingface.co/datasets/hkust-nlp/deita-10k-v0/raw/main/README.md - source-pool description, licence notes, performance table. Fetched 2026-08-11.

[3] Hugging Face Hub API record for hkust-nlp/deita-10k-v0. https://huggingface.co/api/datasets/hkust-nlp/deita-10k-v0?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=hkust-nlp%2Fdeita-10k-v0 Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=hkust-nlp%2Fdeita-10k-v0 Fetched 2026-08-11.

[6] Hugging Face Hub API record for hkust-nlp/deita-10k-v0 with `downloadsAllTime` expansion. https://huggingface.co/api/datasets/hkust-nlp/deita-10k-v0?expand%5B%5D=downloadsAllTime Fetched 2026-08-11.

[7] Hugging Face Hub API dataset listing for author hkust-nlp, filtered to "deita" (https://huggingface.co/api/datasets?author=hkust-nlp&search=deita), plus datasets-server `/size` and `/info` fetched individually for `hkust-nlp/deita-quality-scorer-data`, `hkust-nlp/deita-complexity-scorer-data`, and `hkust-nlp/deita-redundant-pool-data`: https://datasets-server.huggingface.co/size?dataset=<id> and https://datasets-server.huggingface.co/info?dataset=<id> - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[8] The dataset's own served Parquet file, read directly and in full (all 10,000 rows of the `train` split). https://huggingface.co/datasets/hkust-nlp/deita-10k-v0/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet, located via the datasets-server `/parquet` endpoint. Fetched 2026-08-11.

[9] hkust-nlp/deita-6k-v0 dataset card (README), Hub API record, and datasets-server `/info` record. https://huggingface.co/datasets/hkust-nlp/deita-6k-v0/raw/main/README.md , https://huggingface.co/api/datasets/hkust-nlp/deita-6k-v0 , and https://datasets-server.huggingface.co/info?dataset=hkust-nlp%2Fdeita-6k-v0 - `/info` (live, no revision parameter) is the source for the column schema. Fetched 2026-08-11.

[10] The corpus screening row for `hkust-nlp/deita-10k-v0`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT chat training data, with the one caveat established above: the card's prose names WizardLM as a source pool, but no row in the served split carries that source tag, so a reader relying on the prose alone would misdescribe the data's composition. The screening row's note matches the card's own prose description of the source pools.

### The screening row

The row's own note [10]: "10k automatically selected SFT samples drawn from ShareGPT, UltraChat and WizardLM." The row carries no flag.
