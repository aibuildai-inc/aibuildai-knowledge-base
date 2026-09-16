# argilla/ultrafeedback-binarized-preferences-cleaned

60,917 chosen/rejected DPO preference pairs, derived from UltraFeedback, with TruthfulQA-sourced prompts and TruthfulQA-derived ShareGPT prompts removed.

**argilla/ultrafeedback-binarized-preferences-cleaned** is Argilla's re-binarization of the openbmb/UltraFeedback preference dataset introduced by "UltraFeedback: Boosting Language Models with Scaled AI Feedback" [1], built on top of Argilla's own earlier release `argilla/ultrafeedback-binarized-preferences` [2]. The card states it binarizes each UltraFeedback prompt's four rated completions by picking the response with the highest mean of the aspect preference ratings as `chosen` and a lower-mean response at random as `rejected`, rather than using UltraFeedback's `overall_score` field, which the predecessor card documents as bugged for some rows [3][2]. **This release additionally removes every row whose `source` is `truthful_qa`, plus ShareGPT-sourced rows identified by a left join against the `truthful_qa` dataset, after AllenAI flagged TruthfulQA contamination in UltraFeedback; hold out or decontaminate against TruthfulQA-derived evaluation sets before scoring on them regardless** [2]. It serves preference-pair (DPO/reward-model) training, not SFT — the card gives no `chosen`-only SFT split or instruction. It lives at https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned .

**Use it for**: preference-pair training — DPO or a reward model consuming chosen/rejected pairs — with the TruthfulQA-contamination removal already applied, so decontamination work is reduced to whatever new eval sets you add [2]. `chosen` and `rejected` are each a two-turn `[{role, content}]` list starting with the `user` turn and ending with the `assistant` completion; trl's dataset-formats documentation defines this exact shape (a `chosen`/`rejected` pair each containing both turns) as the conversational implicit-prompt preference format, distinct from its recommended explicit-prompt format where `chosen`/`rejected` hold only the assistant turn and the prompt sits in its own field [4]. This dataset already carries a separate `prompt` string column alongside the implicit-prompt `chosen`/`rejected` lists, so a trainer expecting the explicit format can be fed `prompt` directly rather than needing trl's `extract_prompt` helper. See the DPO method card.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tags include `license:mit`), ungated (`"gated": false`, `"private": false`) [5]. No further catch is stated beyond the standard MIT grant.

**Shape**: 60,917 rows, one config (`default`), one split (`train`) [6][7].

**Hold out**: nothing named as a split to hold out — the release has no `test` split [6]. The one contamination risk that must be managed is stated in the opening paragraph above: this card's own removal of TruthfulQA-sourced and TruthfulQA-derived-ShareGPT rows is the fix, not a residual flag, but any TruthfulQA-family evaluation set used downstream should still be checked against `prompt` before scoring [2].

**Origin**: built by Argilla, on top of UltraFeedback's own generations and GPT-4 critique ratings [1][2]. Hub API at the check date: `downloads` 13,197, `downloadsAllTime` 182,403, `likes` 163 [5].

**Trained-on-by**: `argilla/notux-8x7b-v1` — its model card states it is "a preference-tuned version of mistralai/Mixtral-8x7B-Instruct-v0.1 on the argilla/ultrafeedback-binarized-preferences-cleaned dataset using DPO" [8]. `kaist-ai/mistral-orpo-beta`, from the ORPO paper's authors — its model card states it is "fine-tuned exclusively on the 61k instances of the cleaned version of UltraFeedback, argilla/ultrafeedback-binarized-preferences-cleaned" [9]. Argilla's own Notus-7B, despite sharing this dataset's curation lineage, trained on the earlier, non-cleaned `argilla/ultrafeedback-binarized-preferences` per its model card, not on this release [10].

**Introduced by**: no paper — the dataset card [2], building on the UltraFeedback paper [1].

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 60,917 |

One config, `default`, eight columns (datasets-server `/info`, matching the shortlist row's declared columns) [7]:

| column | dtype |
| --- | --- |
| `source` | string |
| `prompt` | string |
| `chosen` | list<struct<content: string, role: string>> |
| `chosen-rating` | float64 |
| `chosen-model` | string |
| `rejected` | list<struct<content: string, role: string>> |
| `rejected-rating` | float64 |
| `rejected-model` | string |

Sizes (datasets-server `/size`) [6]: 143,257,393 bytes of original/Parquet download, 305,744,311 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The predecessor card, which this release inherits its binarization method from, states that UltraFeedback's `overall_score` field is bugged in a way that can rate a low-quality response `10`, discovered by Argilla browsing the data in the Argilla tool and cross-checking against the per-response critique text — this is why the mean-of-preference-ratings method is used instead of `overall_score` [3]. The same card states that this switch is not a minor correction: comparing mean-of-ratings against `overall_score` on the predecessor's roughly 63,000 rows picks a different `chosen` response in about 30,000 of them [3].
- The card states the TruthfulQA-contamination fix was AllenAI's finding, applied here by removing rows with `source=truthful_qa` and by a left join against `truthful_qa` to also catch contaminated ShareGPT rows [2].
- No source states a measured duplicate rate or annotator-agreement figure for this release; GPT-4 supplies the underlying preference ratings, per the origin paper, so there is no separate human-agreement statistic to report [1].
- Of the first 37 served rows (offsets 0-36, `train`), all carry `source="evol_instruct"` — the served rows are grouped by source rather than shuffled at this offset, so this run does not indicate the split's overall source mix and none is claimed [11].

## Load it

```python
import datasets

REV = "770076f077c4c5e298498fa32f804857f46d5134"  # main at the check date
train = datasets.load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", revision=REV, split="train")  # 60,917 rows
```

**Trap**: there is only one split, `train`, and no held-out `test` split is provided by this repository — build your own eval split before training if you need one, since none of these 60,917 rows is pre-reserved [6].

## Neighbors

Every row count below was read live at the check date [12]. This card's release is Argilla's own recommended version over its predecessor: the README states it "is the recommended and preferred dataset by Argilla to use from now on when fine-tuning on UltraFeedback" [2].

- `argilla/ultrafeedback-binarized-preferences` — the direct predecessor, 63,619 rows, one `train` split, still containing TruthfulQA-sourced rows and using a different column schema (`instruction`, `chosen_response`, `rejected_response` as plain strings rather than role/content lists) [2][12].
- `openbmb/UltraFeedback` — the origin dataset this release is derived from, 63,967 rows in one `train` split, unbinarized (each row carries all completions and their ratings rather than a single chosen/rejected pair) [1][12].
- `HuggingFaceH4/ultrafeedback_binarized` — a separate binarization by Hugging Face H4, used to train Zephyr-7B-beta; its card states the "chosen" completion is the response with the highest `overall_score` (the field this card's method avoids), and it retains six splits (`train_prefs`, `train_sft`, `test_prefs`, `test_sft`, `train_gen`, `test_gen`) totaling 187,405 rows. Its card also states that a few hundred mislabeled completions and TruthfulQA-sourced prompts found after Zephyr-7B-beta's release "have been resolved in the latest version of the dataset," and that the older, unfixed revision remains reachable only by pinning an explicit commit; the row count above is read from the current, already-fixed revision [13][12].
- `allenai/ultrafeedback_binarized_cleaned` — AllenAI's TruthfulQA-decontaminated version of `HuggingFaceH4/ultrafeedback_binarized`, its card stating it further removed examples "identified as faulty by Argilla"; six splits totaling 186,421 rows, still using H4's `overall_score`-based chosen/rejected selection rather than this card's mean-of-ratings method [14][12].

## A row

Single served shape (`config="default"`, `split="train"`). From row_idx=0 (datasets-server `/first-rows`) [11], with long code fields truncated:

```json
{
  "source": "evol_instruct",
  "prompt": "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea? Here's some starter code to help you out:\n#include <iostream>\n#include <string>\nusing namespace std;\nint main() {\n    string country;\n    // prompt user for input\n    cout << \"Enter the name of a country: \";\n    cin >> country;\n    // check if country borders the Mediterranean Sea\n    // [C++ code]\n    return 0;\n}",
  "chosen": [
    {"role": "user", "content": "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea? [...]"},
    {"role": "assistant", "content": "Here's a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea:\n\n#include <iostream>\n#include <string>\n#include <set>\n#include <map>\n#include <algorithm>\n\nusing namespace std;\n\nint main() { [...] }"}
  ],
  "chosen-rating": 5.0,
  "chosen-model": "starchat",
  "rejected": [
    {"role": "user", "content": "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea? [...]"},
    {"role": "assistant", "content": "[a lower-rated completion to the same prompt]"}
  ],
  "rejected-rating": 1.25,
  "rejected-model": "pythia-12b"
}
```

## Where it came from

Built by Argilla, layered on the origin UltraFeedback dataset. UltraFeedback itself samples about 64,000 instructions from six public sources (UltraChat, ShareGPT, Evol-Instruct, TruthfulQA, FalseQA, FLAN), queries a pool of 17 models spanning commercial and open-source families for four completions per prompt, and has GPT-4 rate each completion along instruction-following, truthfulness, honesty, and helpfulness [1]. Argilla's predecessor release re-binarized these ratings by taking the mean of the four aspect ratings instead of UltraFeedback's own `overall_score` field, after finding rows where `overall_score` disagreed sharply with the per-response critique text [3][2]. This release starts from that predecessor, removes rows whose `source` is `truthful_qa` and ShareGPT rows matched to `truthful_qa` by a left join, and reformats `chosen`/`rejected` into role/content message lists to match the schema used by `HuggingFaceH4/ultrafeedback_binarized` and `allenai/ultrafeedback_binarized_cleaned` [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Cui et al., "UltraFeedback: Boosting Language Models with Scaled AI Feedback", 2023. https://arxiv.org/abs/2310.01377 — origin dataset construction, current title read from the live abs page. Fetched 2026-08-11.

[2] argilla/ultrafeedback-binarized-preferences-cleaned dataset card (README). https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned/raw/main/README.md — binarization method, TruthfulQA removal, "recommended" claim, schema-alignment rationale. Fetched 2026-08-11.

[3] argilla/ultrafeedback-binarized-preferences dataset card (README), the predecessor this release is built on. https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences/raw/main/README.md — the `overall_score` bug finding and the mean-of-ratings method. Fetched 2026-08-11.

[4] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats — explicit-prompt vs. implicit-prompt preference format definitions. A `main` build, unpinned and mutable. Fetched 2026-08-11.

[5] Hugging Face Hub API record for argilla/ultrafeedback-binarized-preferences-cleaned. https://huggingface.co/api/datasets/argilla/ultrafeedback-binarized-preferences-cleaned?full=true — licence, gate, `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=argilla%2Fultrafeedback-binarized-preferences-cleaned Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=argilla%2Fultrafeedback-binarized-preferences-cleaned Fetched 2026-08-11.

[8] argilla/notux-8x7b-v1 model card (README). https://huggingface.co/argilla/notux-8x7b-v1/raw/main/README.md — states DPO training on this dataset. Fetched 2026-08-11.

[9] kaist-ai/mistral-orpo-beta model card (README). https://huggingface.co/kaist-ai/mistral-orpo-beta/raw/main/README.md — states ORPO training exclusively on this dataset's 61k instances. Fetched 2026-08-11.

[10] argilla/notus-7b-v1 model card (README). https://huggingface.co/argilla/notus-7b-v1/raw/main/README.md — `datasets:` metadata and body text both name the non-cleaned `argilla/ultrafeedback-binarized-preferences` as the training set. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=argilla%2Fultrafeedback-binarized-preferences-cleaned&config=default&split=train — 37 rows returned, offsets 0-36. Fetched 2026-08-11.

[12] datasets-server size endpoint, one call per neighbor: `argilla/ultrafeedback-binarized-preferences`, `openbmb/UltraFeedback`, `HuggingFaceH4/ultrafeedback_binarized`, `allenai/ultrafeedback_binarized_cleaned`. https://datasets-server.huggingface.co/size?dataset=<id> — this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[13] HuggingFaceH4/ultrafeedback_binarized dataset card (README). https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/raw/main/README.md — `overall_score`-based chosen/rejected selection, Zephyr-7B-beta use, split list. Fetched 2026-08-11.

[14] allenai/ultrafeedback_binarized_cleaned dataset card (README). https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned/raw/main/README.md — states it removed examples flagged by Argilla, and is built on top of `HuggingFaceH4/ultrafeedback_binarized`. Fetched 2026-08-11.

[15] The corpus screening row for `argilla/ultrafeedback-binarized-preferences-cleaned`, supplied with this card's request — its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as DPO/preference-pair training data, with no split reserved for held-out evaluation. The card's own removal of TruthfulQA-sourced and TruthfulQA-derived ShareGPT rows, stated in the opening paragraph, is the dataset's answer to the contamination risk the screening row flags below [2].

### The screening row

The row's own note [15]: "cleaned UltraFeedback DPO pairs; Argilla's recommended version, TruthfulQA prompts removed." The row carries no flag.
