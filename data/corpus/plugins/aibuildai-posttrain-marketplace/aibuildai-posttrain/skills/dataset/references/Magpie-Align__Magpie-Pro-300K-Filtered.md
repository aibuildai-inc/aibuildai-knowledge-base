# Magpie-Align/Magpie-Pro-300K-Filtered

300,000 single-turn instruction/response pairs, self-synthesized entirely from Llama-3-70B-Instruct with no human prompts and then filtered down from a 1M-conversation pool - a flagship SFT release of the Magpie project.

**Magpie-Align/Magpie-Pro-300K-Filtered** is built by the Magpie project team (University of Washington and the Allen Institute for AI) and introduced in "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing" [1]. The method exploits the auto-regressive nature of an aligned chat model: feeding it only the left-side template up to the user-turn marker makes it generate its own user query, which is then completed by prompting the same model for the response - no seed instructions, no human writers [1]. The paper used this method on Llama-3-70B-Instruct to generate 4 million such instruction-response pairs, then applied automated quality filters and kept the 300K with the longest responses to form this release [1][2]. It feeds instruction-tuning (SFT) directly; the dataset card also warns not to combine this release with `Magpie-Pro-MT-300K` in the same fine-tuning run because the two share largely the same first turn [2]. It lives at https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered .

**Use it for**: single-turn SFT chat data - each row is one `human`/`gpt` exchange formatted as a `conversations` list, which is the SFT method card's expected multi-turn-capable chat format even though every sampled row here has exactly one turn each way. **Do not fine-tune on this release together with `Magpie-Pro-MT-300K` in the same run - the dataset card says their first turns are largely the same** [2].

**Licence**: `license:llama3` tag on the repository, i.e. Meta's Llama 3 Community License, not an open-source SPDX licence; ungated (`"gated": false`, `"private": false`) [3]. The one catch: this is a data-generation licence, not a code licence - the responses come from Llama-3-70B-Instruct, so downstream use inherits Llama 3's community-license terms on outputs of the model, per the dataset's own licence tag [3]. Note: an automated scan of the README body flagged the substring "MIT," but that is a false match inside ordinary English words ("limit," "limiting," "legitimate," etc.) - the README states no MIT terms anywhere and its only licence declaration is the `llama3` front-matter field [2].

**Shape**: 300,000 rows, one config (`default`), one split (`train`); columns `conversations` (list of `{from, value}`) and `uuid` (string) [3][4].

**Hold out**: nothing - no source flags a decontamination or evaluation-overlap risk. The dataset card is silent on evaluation-set overlap altogether [2], and the origin paper evaluates its fine-tuned models on separate, independent benchmarks (AlpacaEval 2, Arena-Hard, WildBench) rather than on held-out slices of this data [1].

**Origin**: built and released by Magpie-Align (Magpie project authors); both the instructions and the responses are generations from Llama-3-70B-Instruct, filtered by an automated pipeline, with no human-written prompts [1][2]. Hub API at the check date: `downloads` 2,825, `downloadsAllTime` 24,505, `likes` 55 [3].

**Trained-on-by**: the paper's own release, `Magpie-Align/Llama-3-8B-Magpie-Pro-SFT-300K-v0.1`, is Llama-3-8B fine-tuned by SFT on exactly this dataset - its model card lists this dataset under `datasets:` in its metadata and states it is "a fine-tuned version of meta-llama/Meta-Llama-3-8B on Magpie-Align/Magpie-Pro-300K-Filtered dataset" [5]. No third-party (non-Magpie-team) adoption was found in the sources fetched for this card.

**Introduced by**: [1] (Xu, Jiang, Niu, Deng, Poovendran, Choi, Lin).

## Shape

Rows served and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 300,000 |

One config, `default`, with two columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `conversations` | list<struct<from: string, value: string>> |
| `uuid` | string |

No source states token- or sequence-length statistics for this specific 300K-row release; the paper's selection criterion "choose 300K data with the longest responses" [2] implies the responses are the longest in the 1M-pool by design, but no numeric average is given.

Sizes (datasets-server `/size`) [4]: 562,579,366 bytes of Parquet download, 1,026,356,700 bytes decoded in memory.

The row/column/byte counts above come from datasets-server's `/size`, `/info`, and `/first-rows` endpoints, none of which take a revision parameter, so they are live figures for the repository's default branch, not figures pinned to a commit. The Hub API's `sha` for `main` at the check date is `523df96eb7474e97bca6f378b3baa372a4735fcc` [3] - the same commit given in this card's shortlist row and the one pinned in Load it - so these live-endpoint numbers correspond to that commit as of the check date, but would need to be re-fetched to confirm against any later commit.

## Quality

- The README states the filter setup applied to the 1M raw pool before selecting this 300K: input quality rated "average" or higher, an "Instruction Reward" score of -10 or above, removal of repeated or incomplete instructions (e.g. ones ending in ":"), and finally selection of the 300K examples with the longest responses [2].
- The paper defines the reward signal used in filtering as the difference between a reward model's score for the dataset's own response and its score for a base-model response to the same instruction, following the URIAL methodology it cites [1].
- The paper's own ablation is the deciding quality signal: a Llama-3-8B base model fine-tuned only on this 300K-Filtered set scores AlpacaEval 2 LC 25.08% / WR 29.47% against GPT-4-Turbo-1106, LC 52.12% / WR 53.43% against Llama-3-8B-Instruct, and Arena-Hard WR 18.9% - compared with LC 21.65% / WR 22.19% (vs. GPT-4-Turbo-1106) for the unfiltered 300K-Raw subset of the same pool, and LC 22.11% / WR 26.02% for the 200K-Filtered subset - i.e. filtering measurably raises AlpacaEval win rates over the equally-sized raw sample at the same data size [1].
- No source states a measured duplicate-rate or annotator-agreement figure for this release; none is invented here.

## Load it

Pin the revision this card's numbers were read at (the commit given in the shortlist row, matching the Hub API's current `sha` for `main`) [3]:

```python
import datasets

REV = "523df96eb7474e97bca6f378b3baa372a4735fcc"  # main at the check date
train = datasets.load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", revision=REV, split="train")  # 300,000 rows
```

**Trap**: `conversations` is a list column, not flat `prompt`/`response` strings - each element is a `{from, value}` pair with `from` in `{"human", "gpt"}`; a collator built for flat prompt/completion columns will not read this schema without first extracting the two `value` fields.

## Neighbors

Every neighbor below is from the same Magpie-Align org, per the source dataset's own "Dataset Navigation" table [2]; row counts and column counts were read live at the check date [7].

- `Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1` - the full 1,000,000-row raw pool this 300K was filtered from; 23 columns (vs. 2 here), i.e. it retains scoring/metadata fields this release drops [2][7].
- `Magpie-Align/Magpie-Pro-MT-300K-v0.1` - 300,000 rows, 8 columns; the README explicitly warns not to combine it with this release in one fine-tuning run because the two share largely the same first turn [2][7].
- `Magpie-Align/Magpie-Air-300K-Filtered` - the same filter pipeline and row count (300,000, 2 columns) but generated by the smaller Llama-3-8B-Instruct instead of Llama-3-70B-Instruct; the paper reports Air-300K-Filtered scoring lower (AlpacaEval 2 LC 22.66/WR 23.99) than this Pro-300K-Filtered release (LC 25.08/WR 29.47) [1][2][7].
- This corpus prefers Pro-300K-Filtered over the raw pool and over Magpie-Air for SFT, per the paper's own ablation cited above [1].

## A row

One config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8], with the long response truncated:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Can you explain the concept of equitable distribution in family law?"
    },
    {
      "from": "gpt",
      "value": "In family law, equitable distribution refers to the process of dividing marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. [...] 1. **Marital Property**: [...] 2. **Separate Property**: [...] 3. **Equitable Does Not Mean Equal**: [...] [items 4-9 omitted] [...] In summary, equitable distribution is a complex process that aims to divide marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. The court considers various factors to ensure a just and equitable outcome, which may not always result in an equal split."
    }
  ],
  "uuid": "9e215a13-d889-5c29-9230-6852b278197e"
}
```

Of the first 10 served rows, all 10 carry exactly two turns (`human` then `gpt`) [8].

## Where it came from

Built by the Magpie project team. The generation process, described in the paper [1] and restated on the dataset card [2]: the team prompted Llama-3-70B-Instruct with only the left-side chat template up to the position reserved for the user turn, which - because the model is auto-regressive - causes it to generate a plausible user query on its own; the same model is then prompted with that query to produce the response. This produced 4 million raw instruction-response pairs, which the team scored and filtered (input-quality rating, instruction-reward threshold, de-duplication and incompleteness removal, then longest-response selection) down to this 300,000-row set [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Xu, Jiang, Niu, Deng, Poovendran, Choi, Lin, "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing", 2024. https://arxiv.org/abs/2406.08464 - the origin paper; current title read from the live abs page; method description, Table 2 results, reward-difference definition read from the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2406.08464). Fetched 2026-08-11.

[2] Magpie-Align/Magpie-Pro-300K-Filtered dataset card (README). https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered/raw/main/README.md - filter setup, do-not-combine warning, generating model, Dataset Navigation table. Fetched 2026-08-11.

[3] Hugging Face Hub API record for Magpie-Align/Magpie-Pro-300K-Filtered. https://huggingface.co/api/datasets/Magpie-Align/Magpie-Pro-300K-Filtered?full=true - licence tag, gate status, `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Magpie-Align%2FMagpie-Pro-300K-Filtered Fetched 2026-08-11.

[5] Magpie-Align/Llama-3-8B-Magpie-Pro-SFT-300K-v0.1 model card (README). https://huggingface.co/Magpie-Align/Llama-3-8B-Magpie-Pro-SFT-300K-v0.1/raw/main/README.md - `datasets:` metadata field, fine-tuning description. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Magpie-Align%2FMagpie-Pro-300K-Filtered Fetched 2026-08-11.

[7] datasets-server size endpoint, one call per neighbor: `Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1`, `Magpie-Align/Magpie-Pro-MT-300K-v0.1`, `Magpie-Align/Magpie-Air-300K-Filtered`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Magpie-Align%2FMagpie-Pro-300K-Filtered&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn SFT chat data, entirely model-generated with no human prompts, and not to be combined with the sibling `Magpie-Pro-MT-300K` release in the same run per the dataset card's own warning [2]. This rests on the origin paper's method description and the dataset card's filter setup and warning, both already established above [1][2].

### The screening row

The row's own note [row supplied with this card's request]: "The flagship filtered 300k from the Llama-3-70B-Instruct Magpie pool." The row carries no flag.
