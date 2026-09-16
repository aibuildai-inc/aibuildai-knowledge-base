# allenai/tulu-v2-sft-mixture

326,154 instruction-tuning dialogues - a single train split of role/content message lists mixing human-collected conversations with model-generated completions, the SFT data behind the Tulu 2 model suite.

**allenai/tulu-v2-sft-mixture** is the Allen Institute for AI's second-generation supervised-fine-tuning mixture for the Tulu family of assistants, introduced as an improved collection of instruction datasets named for the Tulu-V2 mix in "Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2" [1]. It combines ten component sources into one chat-format dataset: sampled and CoT-augmented FLAN v2, the highest-scoring conversation paths from Open Assistant 1, deduplicated ShareGPT conversations, GPT-4-generated Alpaca and Code-Alpaca completions, LIMA, WizardLM Evol-Instruct, Open-Orca, a small hand-written "hardcoded" identity set, and a science-task subset, each contributing a declared number of examples to the final mix [2]. **The dataset card tags the repository `license: odc-by`, but its own body states that the underlying subsets carry different licenses - Apache 2.0 for FLAN/OASST1/ShareGPT, CC BY-NC 4.0 for GPT4-Alpaca and Code-Alpaca, CC BY-NC-SA for LIMA, no listed license for WizardLM, and MIT for Open-Orca - so parts of the mixture are non-commercial despite the top-level ODC-BY tag, and use of the whole is also bound by the Common Crawl Terms of Use per the card's own license section [2].** It lives at https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture .

**Use it for**: SFT chat training - each row is already an OpenAI-style `messages` list of `{role, content}` turns, so it loads directly into a chat SFT pipeline with no prompt/response extraction step; see the SFT method card. The non-commercial-subset restriction above applies to any commercial use of the mixture as a whole.

**Licence**: `odc-by` (Open Data Commons Attribution) at the repository level, ungated [3]. The catch: several component subsets are individually licensed CC BY-NC 4.0 or CC BY-NC-SA, which the top-level ODC-BY tag does not surface [2].

**Shape**: 326,154 rows, one config (`default`), one split (`train`) [4][5].

**Hold out**: not stated - no source read for this card (the dataset card, the origin paper, or the model cards that trained on it) flags an evaluation-set overlap or names rows to exclude [1][2][6].

**Origin**: built by the Allen Institute for AI (allenai); mixes human-written/human-selected dialogues (OASST1, ShareGPT, LIMA) with completions generated primarily by GPT-4 and other LLMs (GPT4-Alpaca, WizardLM Evol-Instruct, Open-Orca) [2]. Hub API at the check date: 3,253 downloads, 138 likes [3].

**Trained-on-by**: Tulu 2 and Tulu 2 DPO 70B - the `allenai/tulu-2-dpo-70b` model card describes the model as initially fine-tuned on a filtered and preprocessed version of this mixture, before DPO alignment on UltraFeedback [6]. A modified, 2048-token-resplit variant of this mixture (`allenai/tulu-v2-sft-mixture-olmo-2048`) was used to train OLMo-Instruct, per that variant's own card [7].

**Introduced by**: [1] (Ivison et al., "Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2").

## Shape

Splits and rows (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 326,154 |

One config, `default`, three columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `dataset` | string |
| `id` | string |
| `messages` | list\<struct\<role: string, content: string\>\> |

Component subsets and their declared sizes and licenses, as the dataset card states them [2]:

| subset | declared examples | stated license |
| --- | --- | --- |
| FLAN v2 (general sample) | 50,000 | Apache 2.0 |
| FLAN v2 (CoT sample) | 50,000 | Apache 2.0 |
| Open Assistant 1 | 7,708 | Apache 2.0 |
| ShareGPT | 114,046 | "Apache 2.0 listed, no official repo found" |
| GPT4-Alpaca | 20,000 | CC BY-NC 4.0 |
| Code-Alpaca | 20,022 | CC BY-NC 4.0 |
| LIMA | 1,030 | CC BY-NC-SA |
| WizardLM Evol Instruct | 30,000 | "No license provided" |
| Open-Orca | 30,000 | MIT |
| Hardcoded | 140 | (author-written) |
| Science | 7,544 | (unlisted, "standalone release soon") |

These declared per-subset counts sum to 330,490, which does not match the 326,154 rows the repository's own `dataset_info` and the live `/size` endpoint both report [2][4]; no source read for this card reconciles the 4,336-row difference.

Byte sizes: the repository's own `cardData.dataset_info` states `download_size` 554,561,769 bytes and an in-memory `dataset_size` of 1,239,293,363 bytes [2]; the live datasets-server `/size` endpoint agrees on the download size (554,561,769 bytes of parquet) but measures a different in-memory size, 842,317,166 bytes [4] - the two in-memory figures disagree and no source reconciles them. No source states token or sequence-length statistics for this release; the card shows only a length-distribution histogram image, with no axis values given in the text [2].

## Quality

- The card's own "Note" states that some samples contain empty turns, pointing to a GitHub issue titled "Empty message content in tulu-v2-sft-mixture hosted on HF" (`allenai/open-instruct#161`), and says these rows are deliberately kept for reproducibility rather than removed [2][8]. Sampling 160 rows (20 rows each at offsets 0, 50,000, 100,000, 150,000, 200,000, 250,000, 300,000 and 326,000 of the `train` split, spanning `flan_v2`, `cot`, `oasst1`, `sharegpt`, `open_orca`, `science.scierc_relation`, and `hard_coded` rows), none of the 690 message turns checked had empty content [9] - this does not contradict the card's note, since the affected rows may sit outside the offsets sampled.
- The card separately notes that some of the included science examples have "an off-by-one error in the sentence indexing," which the builders describe as having "a small or negligible impact on performance" and say will be corrected in a future release [2].
- No source states a measured duplicate rate or annotator-agreement figure for the mixture as a whole.

## Load it

```python
import datasets

REV = "6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752"  # main at the check date
ds = datasets.load_dataset("allenai/tulu-v2-sft-mixture", revision=REV, split="train")  # 326,154 rows
```

**Trap**: the single `train` split interleaves all ten component subsets with no separate config; the `dataset` column (e.g. `"flan_v2"`, `"sharegpt"`, `"code_alpaca"`) is the only way to isolate a subset, and it must be used to exclude the CC BY-NC-licensed subsets (GPT4-Alpaca, Code-Alpaca, LIMA) before any commercial use of the loaded data [2].

## Neighbors

- `allenai/tulu-v1-sft-mixture` - the predecessor mixture, 489,818 rows, also `odc-by`; the dataset card for this repository links to it as "Tulu V1 Mix" [2], and its own card describes it as "the human data mixture" that this V2 release supersedes [10].
- `allenai/tulu-v2-sft-long-mixture` - 288,554 rows; its own card describes it as a recreation of this mixture that leaves ShareGPT conversations unchunked instead of splitting them into 4096-token pieces, aimed at long-context finetuning [11].
- `allenai/tulu-v2-sft-mixture-olmo-2048` - 380,900 rows; its own card describes it as "a modified version of the Tulu V2 Mix" with long conversations resplit into 2048-token chunks and the hardcoded identity subset replaced with OLMo-specific examples, used to train OLMo-Instruct [7].
- `allenai/tulu-3-sft-mixture` - the successor mixture, 939,343 rows, drawn from a much larger and more multilingual set of `source_datasets` per its own card metadata [12].

No source states a stated preference among these; pick this base mixture unless long-context ShareGPT (long-mixture), OLMo replication (olmo-2048), or the larger multilingual Tulu 3 composition is specifically needed.

## A row

From `config="default"`, `split="train"`, `row_idx=150001` (datasets-server `/rows`) [9], with the first two, longer turns truncated (all other turns in this row are shown in full):

```json
{
  "dataset": "sharegpt",
  "id": "sharegpt_lgEW3ve_0",
  "messages": [
    {
      "role": "user",
      "content": "hi could you pleasse help me, i need to send an email this is the information , I need it to be clear, the email is \"Good afternoon, we would like you to quote for sending fro Ag. Konstantinou , Marousi, Attica, Greece to Qormi, Malta 26, \n• Number of boxes\n14\n• Weight for the whole shipment.\n26,6 kg [...] 3757 (Charger)\n\""
    },
    {
      "role": "assistant",
      "content": "Dear [Name],\n\nI hope this email finds you well. We are interested in obtaining a quote for shipping the following items from Ag. Konstantinou, Marousi, Attica, Greece to Qormi, Malta:\n\n• Number of boxes: 14 [...] Best regards,\n[Your Name]"
    },
    {"role": "user", "content": "thanks"},
    {"role": "assistant", "content": "You're welcome! Let me know if you need any further assistance."},
    {"role": "user", "content": "what would you suggest for the subject of the email?"},
    {"role": "assistant", "content": "For the subject line of the email, you could use something like \"Request for Shipping Quote - Electrical Toothbrushes\" or \"Shipping Inquiry - Ag. Konstantinou to Qormi - Electrical Toothbrushes\". The goal is to make the subject line clear and specific, so that the recipient understands the purpose of the email at a glance."},
    {"role": "user", "content": "thanks"},
    {"role": "assistant", "content": "You're welcome! If you have any further questions, feel free to ask."}
  ]
}
```

This row has 8 turns; other subsets in the same split are shorter - the `flan_v2` rows sampled at offset 0 all have exactly 2 turns (one user, one assistant) [9] - but every row shares the same three-column, role/content-list schema, so one collator serves the whole split.

## Where it came from

Built by the Allen Institute for AI as the training data for the Tulu 2 model suite. The card describes each component's collection method in one line: FLAN v2 and its CoT subset are sampled from the public FLAN v2 mixture; Open Assistant 1 contributes only "the highest-scoring paths in each conversation tree"; ShareGPT is the builders' own processed version of a community-scraped ShareGPT dump; GPT4-Alpaca and Open-Orca are GPT-4-generated completions layered onto existing prompt sets; Code-Alpaca is carried over unchanged from the builders' prior V1 mixture; LIMA is a small curated set taken as-is; WizardLM Evol Instruct is "distilled data of increasing diversity and complexity" subsampled from the public WizardLM release; the hardcoded set is 14 author-written identity prompts repeated ten times each; and the science subset is described as "under development" with a standalone release still pending at the time of writing [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable, which is why Load it pins the revision `6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752` - the commit that was `main` at the check date. That pin covers `load_dataset(..., revision=...)` calls only. The datasets-server `/size`, `/info`, `/rows`, and `/first-rows` endpoints used throughout Shape, Quality, and A row take no revision parameter - passing one to `/size` was tested and had no effect on the response - so every figure and row drawn from those endpoints ([4], [5], [9]) is a live read as of the check date, not a value scoped to the pinned commit, even though it happens to agree with that commit's own `sha` at the time of fetching.

[1] Ivison et al., "Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2", 2023. https://arxiv.org/abs/2311.10702 - the origin paper; current title and abstract read from the live abs page. Fetched 2026-08-11.

[2] allenai/tulu-v2-sft-mixture dataset page and card (README). https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture (item home) and https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture/raw/main/README.md (raw card text) - subset composition, per-subset counts and licenses, license section, empty-turn note, science off-by-one note. Fetched 2026-08-11.

[3] Hugging Face Hub API record for allenai/tulu-v2-sft-mixture. https://huggingface.co/api/datasets/allenai/tulu-v2-sft-mixture?full=true - gate status, `sha`, `downloads`, `likes`, `cardData`. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=allenai%2Ftulu-v2-sft-mixture Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=allenai%2Ftulu-v2-sft-mixture Fetched 2026-08-11.

[6] allenai/tulu-2-dpo-70b model card (README). https://huggingface.co/allenai/tulu-2-dpo-70b/raw/main/README.md - training description ("initially fine-tuned on a filtered and preprocessed" version of this mixture, then DPO on UltraFeedback). Fetched 2026-08-11.

[7] allenai/tulu-v2-sft-mixture-olmo-2048 dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture-olmo-2048/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=allenai%2Ftulu-v2-sft-mixture-olmo-2048 - description as a modified variant used to train OLMo-Instruct; row count 380,900. Fetched 2026-08-11.

[8] GitHub issue allenai/open-instruct#161. https://github.com/allenai/open-instruct/issues/161 - title read from the live issue page ("Empty message content in tulu-v2-sft-mixture hosted on HF"); linked from the dataset card as the source of the empty-turn note. Fetched 2026-08-11.

[9] datasets-server rows endpoint, called at eight offsets (0, 50000, 100000, 150000, 200000, 250000, 300000, 326000) with length=20 each, and the first-rows endpoint at offset 0. https://datasets-server.huggingface.co/rows?dataset=allenai%2Ftulu-v2-sft-mixture&config=default&split=train and https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Ftulu-v2-sft-mixture&config=default&split=train - empty-turn check across 160 rows/690 turns, subset labels seen, and the sampled row shown in "A row". Fetched 2026-08-11.

[10] allenai/tulu-v1-sft-mixture dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/allenai/tulu-v1-sft-mixture/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=allenai%2Ftulu-v1-sft-mixture - predecessor description; row count 489,818. Fetched 2026-08-11.

[11] allenai/tulu-v2-sft-long-mixture dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/allenai/tulu-v2-sft-long-mixture/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=allenai%2Ftulu-v2-sft-long-mixture - recreation description; row count 288,554. Fetched 2026-08-11.

[12] allenai/tulu-3-sft-mixture dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=allenai%2Ftulu-3-sft-mixture - `source_datasets` list; row count 939,343. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT chat-format training data, with the non-commercial-subset caveat carried above: the card's own text mixes an ODC-BY repository tag with several CC BY-NC and CC BY-NC-SA component subsets [2], and the screening row's note describes the mixture's composition in the same terms.

### The screening row

The row's own note: "Tulu-2 SFT mix: FLAN, OASST and ShareGPT human data plus GPT-4-Alpaca, Code-Alpaca and WizardLM Evol synthetic data." The row carries no flag.
