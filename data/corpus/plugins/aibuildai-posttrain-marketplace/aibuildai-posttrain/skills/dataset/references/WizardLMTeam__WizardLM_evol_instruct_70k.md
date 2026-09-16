# WizardLMTeam/WizardLM_evol_instruct_70k

70,000 single-turn instruction/output pairs, each an Alpaca-seed instruction rewritten by ChatGPT through the Evol-Instruct procedure, paired with a ChatGPT-generated response - the training data released alongside the WizardLM paper.

**WizardLMTeam/WizardLM_evol_instruct_70k** is the public release of the instruction-tuning set behind "WizardLM: Empowering large pre-trained language models to follow complex instructions" [1]. The paper starts from Alpaca's 52k seed instructions and runs four epochs of an evolutionary rewriting procedure through the OpenAI ChatGPT API (`gpt-3.5-turbo`), alternating "in-depth" rewrites (added constraints, deepening, concretizing, more reasoning steps, complicated input) and "in-breadth" rewrites, filtering out failed evolutions, to obtain 250k evolved instructions; 70k of those were then sampled - to match the size of Vicuna's 70k ShareGPT training set for a controlled comparison - and ChatGPT was used again to generate the response for each, and that 70k instruction/response set is what this repository serves [1]. In the paper's own head-to-head, a LLaMA-13B model fine-tuned on this 70k set (WizardLM-13b) scores an average of 58.96 across nine automatic benchmarks (MMLU, ARC, HellaSwag, TruthfulQA, HumanEval, GSM8k, AlpacaEval, MT-Bench, WizardEval), against 54.60 for Vicuna-13b (trained on 70k human ShareGPT conversations) and 43.44 for Alpaca-13b (trained on the same 70k-scale Self-Instruct recipe this data's seed instructions came from) [1]. It lives at https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k .

**Use it for**: SFT chat/instruction training on single-turn instruction-response pairs, the SFT method card's shape - `instruction` maps to the prompt, `output` to the target completion; the columns are plain strings with no chat template applied (no system/user/assistant markup).

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated (`"gated": false`, `"private": false`) [2]. The one catch: the README's model-checkpoints table separately lists "Non-commercial" as the licence for the WizardLM-7B/13B/30B-V1.0 checkpoints trained on this kind of data - that label describes those model weights, not this dataset, whose own licence field is MIT [3].

**Shape**: 70,000 rows, one config (`default`), one split (`train`), two string columns (`instruction`, `output`) [4][5].

**Hold out**: nothing. There is a single `train` split; no source among the README, the paper, or the corpus screening row names a held-out test split or documents any overlap between this set and an evaluation benchmark - the paper's own WizardEval test set (218 instructions) is a separately built, manually curated set, distinct from this instruction-tuning data [1][6].

**Origin**: released by the WizardLM team (Microsoft); instructions are ChatGPT rewrites of Alpaca's human/Self-Instruct seed prompts, and responses are ChatGPT (`gpt-3.5-turbo`) generations, with the paper describing the resulting instructions as having almost no direct human involvement in their annotation [1]. Hub API at the check date: `downloads` 2,832, `downloadsAllTime` 33,015, `likes` 198 [2][7].

**Trained-on-by**: the README's checkpoint table links WizardLM-7B-V1.0 directly to this paper as its released model [1][3]. The paper's own Table 1 separately describes fine-tuning a LLaMA-13B model on this same 70k set (reported as "WizardLM-13b"), but the README does not link the Hub checkpoint WizardLM-13B-V1.0 to that paper, so no source confirms that Hub checkpoint is exactly that Table-1 run [1][3]. No further third-party adoption evidence was found in the sources fetched for this card.

**Introduced by**: [1] (Xu et al.).

## Shape

Rows and split (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 70,000 |

One config, `default`, two columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `output` | string |

Sizes (datasets-server `/size`) [4]: 136,750,331 bytes as the original downloaded JSON, 69,165,544 bytes as Parquet, 130,900,545 bytes decoded in memory. No source states per-example sequence-length or token statistics for this 70k release; the paper reports training hyperparameters (max token length 2048, 3 epochs, batch size 4 per GPU) but not a token count for the data itself [1].

## Quality

- No annotator-agreement, duplicate-rate, or contamination-rate figure is stated for this dataset by the README or the paper.
- The paper's construction pipeline includes an "Elimination Evolving" filter step that discards an evolved instruction when: it adds no information gain over the original (judged by ChatGPT), it makes the model unable to respond (heuristically, a response under 80 words containing an apology), the response is only punctuation/stop-words, or the evolved instruction copies phrasing from the evolving prompt itself [1].
- A widely used derivative, `QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered`, states it removed "instances of blatant alignment" from this exact dataset (it names its source as `victor123/evol_instruct_70k`, which now redirects to this repository), leaving 54,974 of the original 70,000 instructions [8]. That derivative's README does not define what it means by "blatant alignment" beyond that phrase; it credits the removal script to a similar cleanup applied to a ShareGPT release [8].

## Load it

Single `train` split, no splitting needed; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-03-10) [2]:

```python
import datasets

REV = "16b48bd8eecad79d4f42e1ab641db317e1b27443"  # main at the check date
train = datasets.load_dataset("WizardLMTeam/WizardLM_evol_instruct_70k", revision=REV, split="train")  # 70,000 rows
```

**Trap**: the repository's file tree lists a single data file, `alpaca_evol_instruct_70k.json`, and the README's own name for the repo predates a later rename - two older aliases, `victor123/evol_instruct_70k` and `WizardLM/WizardLM_evol_instruct_70k`, both now 307-redirect to this repository rather than resolving directly, so a `load_dataset` call against either old id will only work if the client follows the Hub's redirect [9]. There is no `test` split to accidentally merge in.

## Neighbors

Every row count below was read live at the check date [10].

- `WizardLMTeam/WizardLM_evol_instruct_V2_196k` - the same team's larger, later evolved set; the datasets-server reports 143,000 served rows (not the 196k in the name) in a single `train` split, two columns (`idx`, `conversations`) already in a conversational (list-of-turns) shape rather than this repository's flat `instruction`/`output` strings, so it is not schema-compatible with this dataset without reformatting [10]. Its README carries the same boilerplate model table as this repository's and states nothing dataset-specific beyond the licence [11]. Prefer the V2 set for a larger, later-generation Evol-Instruct corpus; prefer this 70k set to reproduce the original WizardLM paper's training setup.
- `QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered` - this exact dataset with 15,026 rows removed for what its README calls "blatant alignment" (undefined beyond that phrase), leaving 54,974 rows, same two string columns, Apache-2.0 licensed [8][10]. It duplicates most of this corpus's rows, so do not mix both into one run.
- `mlabonne/WizardLM_evol_instruct_70k-ShareGPT` - this same 70,000 rows reformatted into a single `conversations` column of ShareGPT-style turn lists, for frameworks that expect that shape rather than flat `instruction`/`output` strings [10].
- The two known old aliases, `victor123/evol_instruct_70k` and `WizardLM/WizardLM_evol_instruct_70k`, are not separate datasets - both now redirect to this repository [9].

## A row

The repository serves one config and one schema. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [12]:

```json
{
  "instruction": "Can you provide a list of healthy habits to maintain a healthy lifestyle? Please format your response as an HTML page with bullet points.\n<html>\n  <body>\n    <h3>Healthy Habits:</h3>\n    <ul>\n      <li>Eating a balanced diet with plenty of fruits and vegetables.</li>\n      <li>Engaging in regular physical activity, such as walking, running, or cycling.</li>\n      <li>Getting enough sleep each night, ideally 7-8 hours.</li>\n      <li>Staying hydrated by drinking plenty of water throughout the day.</li>\n      <li>Limiting alcohol consumption and avoiding smoking.</li>\n      <li>Managing stress through relaxation techniques like meditation or yoga.</li>\n      <li>Regularly visiting a healthcare provider for check-ups and preventative care.</li>\n    </ul>\n  </body>\n</html>",
  "output": "Here's an HTML page with bullet points for healthy habits:\n<html>\n  <body>\n    <h3>Healthy Habits:</h3>\n    <ul>\n      <li>Eating a balanced diet with plenty of fruits and vegetables.</li>\n      <li>Engaging in regular physical activity, such as walking, running, or cycling.</li>\n      <li>Getting enough sleep each night, ideally 7-8 hours.</li>\n      <li>Staying hydrated by drinking plenty of water throughout the day.</li>\n      <li>Limiting alcohol consumption and avoiding smoking.</li>\n      <li>Managing stress through relaxation techniques like meditation or yoga.</li>\n      <li>Regularly visiting a healthcare provider for check-ups and preventative care.</li>\n    </ul>\n  </body>\n</html>"
}
```

This row's evolved instruction already embeds the expected HTML answer inline (a residue of the "complicating input" evolution type, which adds structured input the model must handle); of the first three served rows, instruction/output character lengths run 779/700, 1648/289, and 655/3536, showing the two fields vary independently in length rather than tracking each other [12].

## Where it came from

Built and released by the WizardLM team. The seed pool is Stanford Alpaca's 52k instructions [1]. Each seed instruction is passed through the paper's Evol-Instruct procedure - repeated rounds of ChatGPT-driven "in-depth" rewrites (five prompt types: add constraints, deepen, concretize, increase reasoning steps, complicate input) and "in-breadth" rewrites, applied via the OpenAI ChatGPT API (`gpt-3.5-turbo`), with an elimination filter discarding rewrites that add no information, break the model's ability to respond, or otherwise fail defined criteria - yielding 250k evolved instructions in total across four evolution epochs [1]. To match the 70k size of Vicuna's human-collected ShareGPT training set for a controlled comparison, the authors sampled 70k of the 250k evolved instructions and used ChatGPT again to generate each response, producing the instruction/output pairs this repository serves [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Xu et al., "WizardLM: Empowering large pre-trained language models to follow complex instructions", 2023. https://arxiv.org/abs/2304.12244 - the origin paper; current title read from the live abs page; Evol-Instruct procedure, 250k/70k sampling, elimination filter, training hyperparameters, Table 1 results, checkpoint-paper link. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2304.12244). Fetched 2026-08-11.

[2] Hugging Face Hub API record for WizardLMTeam/WizardLM_evol_instruct_70k. https://huggingface.co/api/datasets/WizardLMTeam/WizardLM_evol_instruct_70k?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date, file tree. Fetched 2026-08-11.

[3] WizardLMTeam/WizardLM_evol_instruct_70k dataset card (README). https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k/raw/main/README.md - one-line description, model-checkpoints licence table (Non-commercial for WizardLM-7B/13B/30B-V1.0), checkpoint-to-paper links. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=WizardLMTeam%2FWizardLM_evol_instruct_70k Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=WizardLMTeam%2FWizardLM_evol_instruct_70k Fetched 2026-08-11.

[6] The corpus screening row for `WizardLMTeam/WizardLM_evol_instruct_70k`, supplied with this card's request - its `note`, read back in the appendix. Checked 2026-08-11.

[7] Hugging Face Hub API record, `downloadsAllTime` variant. https://huggingface.co/api/datasets/WizardLMTeam/WizardLM_evol_instruct_70k?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered dataset card (README) and datasets-server size/info. https://huggingface.co/datasets/QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered/raw/main/README.md ; https://datasets-server.huggingface.co/size?dataset=QuixiAI%2FWizardLM_alpaca_evol_instruct_70k_unfiltered - "blatant alignment" removal, 54,974 remaining rows, source dataset named as `victor123/evol_instruct_70k`, Apache-2.0 licence. Fetched 2026-08-11.

[9] Hugging Face Hub API redirect responses for the two old aliases. https://huggingface.co/api/datasets/victor123/evol_instruct_70k and https://huggingface.co/api/datasets/WizardLM/WizardLM_evol_instruct_70k - both return HTTP 307 with `Location: /api/datasets/WizardLMTeam/WizardLM_evol_instruct_70k`. Fetched 2026-08-11.

[10] datasets-server size and info endpoints, one call per neighbor, for every neighbor row/column count above: `WizardLMTeam/WizardLM_evol_instruct_V2_196k`, `QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered`, `mlabonne/WizardLM_evol_instruct_70k-ShareGPT`. https://datasets-server.huggingface.co/size?dataset=<id> and /info?dataset=<id> - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] WizardLMTeam/WizardLM_evol_instruct_V2_196k dataset card (README). https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k/raw/main/README.md Fetched 2026-08-11.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=WizardLMTeam%2FWizardLM_evol_instruct_70k&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as plain SFT instruction/output data, single `train` split, nothing to hold out. The dataset's own MIT licence carries no usage-shape restriction beyond attribution, the response and instruction text are both ChatGPT-era generations rather than human-authored dialogue, and the screening row's own note - "Evol-Instruct instructions with ChatGPT-era answers" - matches what the paper and card describe [1][3][6].

### The screening row

The row's own note [6]: "Evol-Instruct instructions with ChatGPT-era answers." The row carries no flag.
