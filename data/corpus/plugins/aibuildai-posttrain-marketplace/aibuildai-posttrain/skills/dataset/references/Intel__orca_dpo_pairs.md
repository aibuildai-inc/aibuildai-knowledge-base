# Intel/orca_dpo_pairs

12,859 DPO preference pairs built from the Open-Orca/OpenOrca dataset, pairing a ChatGPT-generated reply against a Llama-2-13B-chat reply for the same FLAN-sourced prompt, in a single `train` split.

**Intel/orca_dpo_pairs** was built by Intel from [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), which the dataset's own README describes as "12k examples from Orca style dataset Open-Orca/OpenOrca" [1], with "Orca" pointing to "Orca: Progressive Learning from Complex Explanation Traces of GPT-4" [2]. OpenOrca is itself a collection of FLAN Collection prompts augmented with GPT-3.5/GPT-4 reasoning-trace completions [3]. Each row here pairs a `question` (and optional `system` prompt) with a `chosen` and a `rejected` reply; at the commit immediately before this card's pinned revision, those two reply columns were named `chatgpt` and `llama2-13b-chat` rather than `chosen`/`rejected`, and the file's own content is unchanged across the rename [4][5] - so the chosen reply is a ChatGPT completion and the rejected reply is a Llama-2-13B-chat completion, and the "chosen" label was assigned by construction, not by an independent judge. **A community reformatting of this same data re-judged every pair with GPT-4-turbo and found the chosen (ChatGPT) reply was not the better one in about 2,000 of the 12,859 pairs** [6]; use that fact to decide whether to filter this release or use the judged alternative described in Neighbors. It lives at https://huggingface.co/datasets/Intel/orca_dpo_pairs .

**Use it for**: the DPO/preference-pair training shape - `chosen` and `rejected` are already separated from the shared prompt (`question`, plus optional `system`), matching what TRL's dataset-formats guide calls the explicit-prompt preference type, a prompt field alongside chosen and rejected completions, rather than a format with the prompt folded into each completion [7]. The column names are not TRL's `prompt`/`chosen`/`rejected` schema, though: concatenate `system` and `question` into a `prompt` field first. See the DPO method card. Not annotated for SFT use; the card states no such restriction, but the pairs are model-vs-model completions with a construction-assigned label, not a curated instruction set.

**Licence**: Apache-2.0 (`cardData.license` is `apache-2.0`, tagged `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [8]. The one catch: the underlying OpenOrca data this release is built from carries a different license, MIT, under Open-Orca's own dataset card [3]; neither card states why the two licenses differ.

**Shape**: 12,859 rows, one config (`default`), one split `train`, four string columns (`system`, `question`, `chosen`, `rejected`) [9][10].

**Hold out**: nothing found in the sources checked. A community decontamination pass over this same row set matched the `question` text against gsm8k's `train` and `test` splits with TF-IDF cosine similarity at a 0.8 threshold: it reports finding no matches against the gsm8k test split, and 79 of 12,859 questions (0.6%) matching the gsm8k train split [6]. Train-train duplication against another dataset's train split is not an evaluation-holdout risk by itself, and no source here names an overlap with an evaluation set.

**Origin**: built by Intel; both candidate replies are model-generated (ChatGPT and Llama-2-13B-chat), with the "chosen" side determined by which model produced it rather than by a preference judgment at construction time [1][4][5]. Hub API as of the check date: `downloads` 2,148, `downloadsAllTime` 119,343, `likes` 324 [8].

**Trained-on-by**: Intel's own `neural-chat-7b-v3-1`, whose model card states it was aligned with DPO using this dataset [11]. `mlabonne/NeuralHermes-2.5-Mistral-7B` trained on `mlabonne/chatml_dpo_pairs` [12], a ChatML-reformatted copy of this same dataset, confirmed to carry the same 12,859 rows by a live datasets-server check, whose own card describes itself as a preprocessed version of this dataset [13][14] - so NeuralHermes trained on this data through that reformatting, not on the raw Intel repository directly.

**Introduced by**: no paper - the dataset card [1], built from Open-Orca/OpenOrca [3], which cites the Orca paper [2] as its source methodology.

## Shape

Rows and splits (datasets-server `/size`) [9]:

| split | rows |
| --- | --- |
| `train` | 12,859 |
| total | 12,859 |

One config, `default`, four columns, all strings (datasets-server `/info`) [10]:

| column | dtype |
| --- | --- |
| `system` | string |
| `question` | string |
| `chosen` | string |
| `rejected` | string |

Sizes (datasets-server `/size`) [9]: 36,310,081 bytes of original JSON-Lines download, 19,093,564 bytes as Parquet, 34,684,414 bytes decoded in memory. No source states sequence-length or token statistics for this release.

## Quality

- Both `chosen` and `rejected` are model completions, not human writing: at the commit before this card's pinned revision, the two reply columns were literally named `chatgpt` and `llama2-13b-chat`, and Intel's rename to `chosen`/`rejected` at the pinned commit left the row content unchanged, confirmed by comparing row 0 of both revisions [4][5]. The dataset's own README states nothing about how "chosen" was decided; the column names themselves are the only evidence, and they show it was assigned by which model produced the reply rather than by a preference judgment.
- A community reformatting of this same 12,859-row set (`argilla/distilabel-intel-orca-dpo-pairs`) re-judged every pair with a GPT-4-turbo rater and reports: about 4,000 pairs rated a tie, about 7,000 pairs where the original `chosen` (ChatGPT) reply was confirmed better, and about 2,000 pairs where the rater preferred the original `rejected` (Llama-2-13B-chat) reply instead - i.e., roughly 2,000 of 12,859 (about 16%) of this release's chosen/rejected labels are contradicted by that independent judge [6].
- The same reformatting's decontamination pass found 0 of 12,859 questions matching gsm8k's test split and 79 (0.6%) matching gsm8k's train split, using TF-IDF cosine similarity at a manually-checked 0.8 threshold [6].
- Of the first 67 served rows, 62 (93%) carry a non-empty `system` field; the remaining 5 have an empty string [15].
- Intel's README states nothing about annotator process, since there is no human annotation step - both sides are model output [1].

## Load it

```python
import datasets

REV = "624952e3f420ae18d88b31977ee2ea436c833abb"  # main at the check date; commit titled "revise fields name."
train = datasets.load_dataset("Intel/orca_dpo_pairs", revision=REV, split="train")  # 12,859 rows
```

**Trap**: the repository has one split, `train`, and no held-out evaluation split of its own - if you need a held-out portion, carve it out yourself before training, since the source Orca/OpenOrca pipeline this data comes from is not a fixed benchmark test set that decontamination tooling elsewhere already accounts for [1][3]. Also, `chosen` and `rejected` here are not what a GPT-4-turbo judge would pick about 16% of the time per the community re-rating in Quality [6]; if that matters for your use case, filter or switch to the judged neighbor below rather than assuming this release's labels are all correct.

## Neighbors

- `argilla/distilabel-intel-orca-dpo-pairs` - the same 12,859 rows, re-judged with a GPT-4-turbo rater via the `distilabel` library, expanded to 16 columns including `status` (`unchanged`/`tie`/`swapped`), `chosen_score`, ratings, rationales, and an `in_gsm8k_train` flag; about 2,000 rows have `chosen`/`rejected` swapped relative to this release, and the card recommends filtering to non-tie, `chosen_score >= 8`, non-gsm8k-train rows, which yields 5,922 rows and reportedly better downstream performance than training on the full original [6][16]. Prefer this over the raw Intel release when the judged label or the gsm8k-train flag matters.
- `mlabonne/chatml_dpo_pairs` - this same dataset, reformatted into ChatML `prompt`/`chosen`/`rejected` strings (system+question folded into `prompt`, `<|im_end|>` appended to replies) [13], and confirmed to carry the same 12,859 rows by a live datasets-server check [14]. Convenience reformatting only; it carries no additional filtering or judgment beyond this release.

## A row

One served split, one shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [15], with the long `rejected` explanation truncated:

```json
{
  "system": "",
  "question": "You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:",
  "chosen": "[\n  [\"AFC Ajax (amateurs)\", \"has ground\", \"Sportpark De Toekomst\"],\n  [\"Ajax Youth Academy\", \"plays at\", \"Sportpark De Toekomst\"]\n]",
  "rejected": " Sure, I'd be happy to help! Here are the RDF triplets for the input sentence:\n\n[AFC Ajax (amateurs), hasGround, Sportpark De Toekomst]\n[Ajax Youth Academy, playsAt, Sportpark De Toekomst]\n\nExplanation:\n\n* AFC Ajax (amateurs) is the subject of the first triplet [...] but the above triplets capture the main relationships present in the sentence."
}
```

## Where it came from

Intel built this repository from Open-Orca/OpenOrca [1], a collection of FLAN Collection prompts each submitted to GPT-3.5 or GPT-4 for an augmented reasoning-trace completion, following the method of "Orca: Progressive Learning from Complex Explanation Traces of GPT-4" [2][3]. For this DPO release, Intel additionally generated a second completion per prompt from Llama-2-13B-chat and packaged the two as a chosen/rejected pair, ChatGPT as chosen and Llama-2-13B-chat as rejected - visible directly in the repository history, where the reply columns were named `chatgpt` and `llama2-13b-chat` at the commit before this card's pinned revision and were renamed to `chosen`/`rejected`, with identical row content, at the pinned commit itself [4][5].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Intel/orca_dpo_pairs dataset card (README). https://huggingface.co/datasets/Intel/orca_dpo_pairs/raw/main/README.md - full card text, license field. Fetched 2026-08-12.

[2] Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4", 2023. https://arxiv.org/abs/2306.02707 - the paper the Intel README links as "Orca"; current title read from the live abs page. Fetched 2026-08-12.

[3] Open-Orca/OpenOrca dataset card (README). https://huggingface.co/datasets/Open-Orca/OpenOrca/raw/main/README.md - MIT license, FLAN Collection sourcing, GPT-3.5/GPT-4 augmentation description. Fetched 2026-08-12.

[4] Hugging Face Hub commit history for Intel/orca_dpo_pairs. https://huggingface.co/api/datasets/Intel/orca_dpo_pairs/commits/main - the pinned commit's own title, "revise fields name.", and the preceding commit id used to read the pre-rename schema. Fetched 2026-08-12.

[5] Row 0 of `orca_rlhf.jsonl` at the commit before the pinned revision (`af849bff38b022f0aa211a0f5a34d08a155c3a32`). https://huggingface.co/datasets/Intel/orca_dpo_pairs/resolve/af849bff38b022f0aa211a0f5a34d08a155c3a32/orca_rlhf.jsonl - shows the `chatgpt`/`llama2-13b-chat` column names and content, compared against row 0 at the pinned commit (source [15]) to confirm only the column names changed. Fetched 2026-08-12.

[6] argilla/distilabel-intel-orca-dpo-pairs dataset card (README). https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs/raw/main/README.md - GPT-4-turbo re-judging results (tie/unchanged/swapped counts), gsm8k train/test decontamination method and result (0 test matches, 79/12,859 train matches), recommended filter and resulting row count. Fetched 2026-08-12.

[7] TRL dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats - defines the explicit-prompt preference type as a `prompt` field alongside `chosen`/`rejected` completions, distinct from the implicit-prompt type where the prompt is folded into each completion. A `main` build, unpinned and mutable. Fetched 2026-08-12.

[8] Hugging Face Hub API record for Intel/orca_dpo_pairs. https://huggingface.co/api/datasets/Intel/orca_dpo_pairs?full=true and the same endpoint with `expand[]=downloadsAllTime` - license, gate, `sha`, `downloads`, `downloadsAllTime`, `likes`, `lastModified`. Fetched 2026-08-12.

[9] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Intel%2Forca_dpo_pairs Fetched 2026-08-12.

[10] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Intel%2Forca_dpo_pairs Fetched 2026-08-12.

[11] Intel/neural-chat-7b-v3-1 model card (README). https://huggingface.co/Intel/neural-chat-7b-v3-1/raw/main/README.md - states the model was aligned using DPO with this dataset. Fetched 2026-08-12.

[12] mlabonne/NeuralHermes-2.5-Mistral-7B model card (README). https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B/raw/main/README.md - states DPO fine-tuning used the `mlabonne/chatml_dpo_pairs` dataset. Fetched 2026-08-12.

[13] mlabonne/chatml_dpo_pairs dataset card (README) and preprocessing code. https://huggingface.co/datasets/mlabonne/chatml_dpo_pairs/raw/main/README.md - describes the dataset as a preprocessed version of Intel/orca_dpo_pairs; its code reads `example['chatgpt']` and `example['llama2-13b-chat']`, corroborating the pre-rename column names in source [5]. Fetched 2026-08-12.

[14] datasets-server size endpoint for mlabonne/chatml_dpo_pairs. https://datasets-server.huggingface.co/size?dataset=mlabonne%2Fchatml_dpo_pairs - row count (12,859); this endpoint takes no revision parameter, so this count is live, not pinned. Fetched 2026-08-12.

[15] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Intel%2Forca_dpo_pairs&config=default&split=train - 67 served rows read for the sampled row and the `system`-field non-emptiness count. Fetched 2026-08-12.

[16] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=argilla%2Fdistilabel-intel-orca-dpo-pairs - row count (12,859) and column count (16); this endpoint takes no revision parameter, so this count is live, not pinned. Fetched 2026-08-12.

[17] The corpus screening row for `Intel/orca_dpo_pairs`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as DPO preference-pair training data, with the caveat established above: about 16% of this release's chosen/rejected labels are contradicted by an independent GPT-4-turbo judge in the same rows reformatted elsewhere, so a filtered or re-judged variant may be preferable depending on tolerance for label noise [6]. No evaluation-set holdout requirement was found in any source checked, appendix included.

### The screening row

The row's own note [17]: "12k OpenOrca items (FLAN prompts, GPT-4 answers) turned into DPO pairs." The row carries no flag.
