# dmis-lab/meerkat-instructions

439,834 served chat-format instruction-tuning rows across eight medical QA and dialogue configs - a mix of GPT-4-written reasoning, GPT-3.5-cleaned or -simulated dialogue, and unmodified human-authored QA pairs, all as `id`/`messages` rows.

**dmis-lab/meerkat-instructions** is the instruction-tuning data behind Korea University DMIS Lab's Meerkat medical language models, released alongside "Small language models learn enhanced reasoning skills from medical textbooks" [1]. It lives at https://huggingface.co/datasets/dmis-lab/meerkat-instructions . Nine configs are declared in the repository's YAML, covering multiple-choice QA, free-form single-turn QA, multi-turn dialogue, and clinical-note generation [2], but the ninth config, MTS-dialog, fails to load through the Hub's dataset-server: its declared data-file path carries a leading space that does not match the actual file name in the repository, so its 1,200 rows are unreachable through `load_dataset` even though the underlying file (`mts-dialog.jsonl`) is present in the repo tree [2][3][4]. **The card does not state which split of MedQA or MedMCQA feeds the MedQA-CoT, MedQA-dialog, and MedMCQA configs; row-count matching against each source's own Hub copy shows the MedMCQA config's 182,822 rows equal that dataset's train split exactly, and MedQA-CoT's 9,308 rows are a subset of MedQA's 10,178-row train split, so neither appears to reuse the paper's own MedQA or MedMCQA test benchmarks - but this is inferred from row counts, not stated on the card, so decontaminate against MedQA and MedMCQA test/benchmark splits before scoring a model trained on this data against those benchmarks [2][5][6].**

**Use it for**: reasoning-trace and general instruction SFT in chat format - in the first-rows page fetched from each config (100 rows for ChatDoctor-cleaned, LiveQA, MedInstruct-52K, MedMCQA, and MedicationQA; 76 for MedBooks-18-CoT; 65 for MedQA-CoT; 62 for MedQA-dialog), the seven configs other than MedQA-dialog are single-turn (system/user/assistant) triples and MedQA-dialog is multi-turn dialogue with a variable number of turns [7]; **non-commercial use only** (cc-by-nc-4.0), and decontaminate against MedQA/MedMCQA test sets before benchmark scoring as above. Rows are already `messages` lists with `role`/`content` pairs, so they map directly onto the SFT method card's chat format with no prompt/response extraction step needed.

**Licence**: cc-by-nc-4.0, ungated [2][3]. The catch: this is a non-commercial licence on the compiled repository, and the card gives no licence information for the underlying MedQA, MedMCQA, LiveQA, MedicationQA, ChatDoctor, or AlpaCare/MedInstruct source data it repackages [2].

**Shape**: 9 configs declared, 8 load through the Hub's dataset-server (MTS-dialog is broken, see above); each of the 8 is a single `train` split; 439,834 rows served in total, `id` (string) and `messages` (list of `{role, content}` structs) columns [2][3][4].

**Hold out**: nothing within this dataset itself - it declares no eval split. The risk is external: decontaminate against the MedQA and MedMCQA test benchmarks (1,273 and 6,150 rows respectively in their standard Hub releases) before scoring a model trained on this data against those benchmarks, since this dataset draws on the same MedQA/MedMCQA question pools [1][2][5][6].

**Origin**: built by DMIS Lab (Korea University); reasoning traces in MedQA-CoT and MedBooks-18-CoT are GPT-4 generations, ChatDoctor-cleaned and MedQA-dialog are GPT-3.5 generations/cleanings, and LiveQA, MedicationQA, MedMCQA, and MedInstruct-52K are reused from their original human-authored sources without modification [2]. Hub API at the check date: `downloads` 744, `downloadsAllTime` 4,082, `likes` 10 [3].

**Trained-on-by**: the paper's own Meerkat-7B, Meerkat-8B, and (in a follow-up experiment) a Llama-3-70B-based Meerkat-70B, all fine-tuned on this data, which the dataset's own card describes as the instruction-tuning data used to train those models [1][2]. Meerkat-7B and Meerkat-8B reached average accuracies of 64.5% and 66.7% across six medical QA benchmarks, ahead of Mistral-7B (41.2%), Llama-3-8B (56.1%), MediTron-7B (51.0%), BioMistral-7B (55.4%), and GPT-3.5 (54.8%) [1]; in a follow-up experiment, Meerkat-70B - a Llama-3-70B trained on this same data - outperformed the plain Llama-3-70B baseline by 2.9 points averaged across the same six benchmarks, and surpassed GPT-4 and GPT-4o by 1.4 and 0.5 points respectively [1]. No adoption by other released models or recipes found.

**Introduced by**: [1] (Kim et al., *npj Digital Medicine*, 2025).

## Shape

Rows and columns per config, from the datasets-server `/size` and `/info` endpoints [4][5]; all 8 loadable configs share one `train` split and the `id`/`messages` schema:

| config | # rows served | # rows on card |
| --- | --- | --- |
| `ChatDoctor-cleaned` | 111,902 | 111,902 |
| `LiveQA` | 633 | 633 |
| `MedBooks-18-CoT` | 77,660 | 77,660 |
| `MedInstruct-52K` | 52,002 | 52,002 |
| `MedMCQA` | 182,822 | 182,822 |
| `MedQA-CoT` | 9,308 | 9,308 |
| `MedQA-dialog` | 4,818 | 4,818 |
| `MedicationQA` | 689 | 689 |
| `MTS-dialog` | 0 (fails to load) | 1,200 |
| **Total** | **439,834** | **441,034** |

The card's own statistics table states the total as 441,034 rows across all nine configs [2]; the 1,200-row gap against the 439,834 rows the dataset-server actually serves equals MTS-dialog's declared row count exactly, matching the broken-path diagnosis above [2][4].

Both columns are present in every loadable config: `id` (string) and `messages` (a list of `{role: string, content: string}` structs) [5]. No source states sequence-length or token statistics for any config; none is invented here.

Byte sizes for the whole repository (datasets-server `/size`) [4]: 569,439,913 bytes of original JSONL, 227,431,512 bytes as Parquet, 531,692,523 bytes decoded in memory.

## Quality

- MedQA-CoT and MedBooks-18-CoT reasoning traces are GPT-4-generated chains of thought over MedQA training questions and questions synthesized from 18 medical textbooks respectively; the card gives no measured accuracy or error rate for these traces [2].
- ChatDoctor-cleaned is GPT-3.5-filtered from the 112K-pair ChatDoctor HealthCareMagic-100k set, with greetings and sign-offs stripped by GPT-3.5 given manually written in-context examples, yielding 111,901 cleaned pairs on the card; the card attributes the served total's small offset from that figure (111,902 here) to an unspecified number of samples dropped for GPT-3.5 inference errors during cleaning, without giving a count [2].
- MedQA-dialog is GPT-3.5-simulated multi-turn patient-doctor dialogue built from MedQA questions and their MedQA-CoT answers, with the assistant role instructed to ask follow-up questions when information is missing [2]. Of the first 62 served rows, message-list length ranges from 3 to 27, so single-exchange and long multi-turn dialogues both occur [7].
- LiveQA, MedicationQA, and MedMCQA are stated to be used unmodified from their original sources, with no GPT relabeling [2]; no source states an annotation-agreement or error-rate figure for them.
- No source states an overall duplicate-row or near-duplicate rate for the compiled repository.

## Load it

Only the eight configs that load will return rows; MTS-dialog raises a file-not-found error. Two different revisions are in play here, and they are not the same commit: the Hub API's current `sha` for `main` at the check date is `f0621a0b2eb9f6c48d6ec5299aecc03d6d350afc` [3], but the row counts and byte sizes reported throughout this card come from the datasets-server endpoints (`/size`, `/info`, `/first-rows`), whose own `download_checksums` name the commit they actually processed as `348baeee8e8470dd4551de47726e15a16cec9fbf` - an older commit, not `f0621a0b...` [5]. Those endpoints take no `revision` query parameter: appending `&revision=f0621a0b2eb9f6c48d6ec5299aecc03d6d350afc` to the `/size` call returns byte-identical output to the unpinned call, so datasets-server cannot be re-pinned on demand and this card cannot independently confirm that `348baeee...` and current `main` agree in content [4]. `load_dataset(..., revision=REV)` below pins to current `main`; it is not guaranteed to reproduce every row count on this card if the repository changed between `348baeee...` and `f0621a0b...`:

```python
import datasets

REV = "f0621a0b2eb9f6c48d6ec5299aecc03d6d350afc"  # main at the check date; datasets-server's own cached figures on this card reflect commit 348baeee8e8470dd4551de47726e15a16cec9fbf instead
medmcqa = datasets.load_dataset("dmis-lab/meerkat-instructions", "MedMCQA", revision=REV, split="train")        # 182,822 rows per datasets-server
medqa_dialog = datasets.load_dataset("dmis-lab/meerkat-instructions", "MedQA-dialog", revision=REV, split="train")  # 4,818 rows per datasets-server, multi-turn
```

**Trap**: `datasets.load_dataset("dmis-lab/meerkat-instructions", "MTS-dialog", ...)` fails - the YAML `data_files` path for this config is `" mts-dialog.jsonl"` (leading space), which does not match the actual file `mts-dialog.jsonl` in the repo tree, so this config cannot be loaded through the Hub loader at all; the file must be pulled directly (e.g. via `hf_hub_download`) and parsed as JSONL to recover its declared 1,200 rows [2][3].

## Neighbors

No re-release or successor of this exact compiled repository was found on the Hub. Several of its component sources exist independently and can be preferred over the corresponding config here when the GPT-written reformatting is not wanted:

- `lavita/AlpaCare-MedInstruct-52k` - 52,002 rows, matching the `MedInstruct-52K` config's row count exactly; this is the unmodified AlpaCare instruction set this config repackages into `messages` format [8][9].
- `lavita/ChatDoctor-HealthCareMagic-100k` - 112,165 rows, close to but not identical to the 111,902 in `ChatDoctor-cleaned` (the card's stated GPT-3.5 cleaning step removed a small, uncounted number of samples); use this raw form if the uncleaned greetings/sign-offs are acceptable or preferred for a different cleaning pass [2][8].
- `openlifescienceai/medmcqa` and `GBaker/MedQA-USMLE-4-options` - the original MedMCQA and MedQA releases this repository's `MedMCQA` and `MedQA-CoT`/`MedQA-dialog` configs draw from, each with their own declared train/test/validation splits, useful for confirming the decontamination boundary above directly rather than through row-count inference [5][6].

This corpus prefers this compiled repository over its component sources when the chat-formatted `messages` shape is wanted directly, since it saves the reformatting step each of the above would otherwise require.

## A row

Across the first-rows page fetched from each config (100 rows for ChatDoctor-cleaned, LiveQA, MedInstruct-52K, MedMCQA, and MedicationQA; 76 for MedBooks-18-CoT; 65 for MedQA-CoT; 62 for MedQA-dialog), two distinct shapes appear: single-turn triples in the seven configs other than MedQA-dialog, and multi-turn dialogue in MedQA-dialog [7]. From `config="MedMCQA"`, `split="train"`, `row_idx=0` [7]:

```json
{
  "id": "medmcqa_0",
  "messages": [
    {"role": "system", "content": "Answer the multiple-choice question about medical knowledge."},
    {"role": "user", "content": "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma (A) Hyperplasia (B) Hyperophy (C) Atrophy (D) Dyplasia"},
    {"role": "assistant", "content": "The answer is (C) Atrophy."}
  ]
}
```

From `config="MedQA-dialog"`, `split="train"`, `row_idx=0`, with the system prompt and final assistant turn truncated [7]:

```json
{
  "id": "medqa-dialog_0",
  "messages": [
    {"role": "system", "content": "You are a helpful doctor or healthcare professional. Guide the conversation to provide useful, complete, and scientifically-grounded answers to user questions. [...] you should engage in a multi-turn dialogue, asking follow-up questions [...]"},
    {"role": "user", "content": "I'm 23 years old and currently 22 weeks pregnant. I've been experiencing a burning sensation when I urinate for the past day. [...] My temperature is 97.7°F, blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98%."},
    {"role": "assistant", "content": "I'm sorry to hear that you're experiencing discomfort. Based on your symptoms, it seems like you may be dealing with a urinary tract infection (UTI) [...] may I ask if you have any other symptoms such as fever or pain in the back area?"},
    {"role": "user", "content": "No, I don't have any fever or pain in my back. My physical exam also showed no tenderness in the costovertebral angle, and my uterus is gravid."},
    {"role": "assistant", "content": "Thank you for sharing that additional information. [...] Do you have any concerns about using antibiotics during your pregnancy?"},
    {"role": "user", "content": "I do have some concerns, but I want to do what's best for both me and my baby. I want to ensure the medication is safe."},
    {"role": "assistant", "content": "Absolutely, it's crucial to prioritize safety. [...] Based on your current gestational age of 22 weeks, nitrofurantoin would likely be the best choice. [...] So, in your case, the best treatment option would be Nitrofurantoin."}
  ]
}
```

## Where it came from

Built and released by DMIS Lab (Korea University) as the training data for its Meerkat model family [1][2]. The repository combines newly generated data with reused human-authored sources [2]:

- **MedQA-CoT**: GPT-4-generated chain-of-thought reasoning over questions from the MedQA training set [2].
- **MedBooks-18-CoT**: questions generated from 18 medical textbooks, paired with GPT-4-generated chain-of-thought reasoning [2].
- **ChatDoctor-cleaned**: derived from the ChatDoctor HealthCareMagic-100k set of 112K real online medical-consultation Q&A pairs, with greetings and sign-offs removed by GPT-3.5 using manually written in-context examples [2].
- **MedQA-dialog**: GPT-3.5-simulated multi-turn patient-doctor conversations built from MedQA questions and their MedQA-CoT answers, with the assistant role prompted to ask follow-up questions when information is missing [2].
- **LiveQA**, **MedicationQA**, **MedMCQA**, and **MedInstruct-52K** are used in their original, unmodified form from their respective source papers [2].
- **MTS-dialog** is used in its original form but is not reachable through the Hub loader due to the path defect described above [2][3][4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Kim, H. et al., "Small language models learn enhanced reasoning skills from medical textbooks", *npj Digital Medicine*, vol. 8, no. 1, p. 240, 2025. https://www.nature.com/articles/s41746-025-01653-8 - origin paper: benchmark table stating MedMCQA benchmark is the MedMCQA test split, training method, and Meerkat-7B/8B/70B training. Fetched 2026-08-11.

[2] dmis-lab/meerkat-instructions dataset card (README), including its YAML config block. https://huggingface.co/datasets/dmis-lab/meerkat-instructions/raw/main/README.md - config declarations and paths, statistics table, per-dataset descriptions, licence field, references. Fetched 2026-08-11.

[3] Hugging Face Hub API record for dmis-lab/meerkat-instructions. https://huggingface.co/api/datasets/dmis-lab/meerkat-instructions?full=true - licence tag, gate status, `sha`, `cardData.configs` (including the MTS-dialog path with its leading space), the `siblings` file list (which includes `mts-dialog.jsonl` with no leading space), `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=dmis-lab%2Fmeerkat-instructions - per-config and total row/byte counts; MTS-dialog absent from the config list. Re-called with `&revision=f0621a0b2eb9f6c48d6ec5299aecc03d6d350afc` appended, which returned byte-identical output to the unpinned call, showing the endpoint ignores this parameter. Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=dmis-lab%2Fmeerkat-instructions - per-config feature schema; MTS-dialog absent; `download_checksums` names the processed commit as `348baeee8e8470dd4551de47726e15a16cec9fbf`, distinct from the Hub API's current `main` `sha`. Fetched 2026-08-11.

[6] Original source-dataset row counts, read live to check split alignment: `openlifescienceai/medmcqa` (train 182,822 / test 6,150 / validation 4,183) and `GBaker/MedQA-USMLE-4-options` (train 10,178 / test 1,273), via https://datasets-server.huggingface.co/size?dataset=<id>. These endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[7] datasets-server first-rows endpoint, one call per config, for the message-length ranges and the two sampled rows above: https://datasets-server.huggingface.co/first-rows?dataset=dmis-lab%2Fmeerkat-instructions&config=<config>&split=train, called for all eight loadable configs. Fetched 2026-08-11.

[8] Neighbor dataset row counts, read live: `lavita/AlpaCare-MedInstruct-52k` (52,002 rows) and `lavita/ChatDoctor-HealthCareMagic-100k` (112,165 rows), via https://datasets-server.huggingface.co/size?dataset=<id>. Live, not pinned. Fetched 2026-08-11.

[9] Hugging Face Hub API listing of dmis-lab's public datasets, used to check for a sibling or successor release of this repository. https://huggingface.co/api/datasets?author=dmis-lab&limit=100 - none found with overlapping content. Fetched 2026-08-11.

[10] The corpus screening row for `dmis-lab/meerkat-instructions`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT data with two caveats already established above: it is non-commercial (cc-by-nc-4.0), and the MedQA/MedMCQA-derived configs' split is not stated on the card - row-count matching against each source's own train split suggests no overlap with the paper's own MedQA/MedMCQA test benchmarks, but a model trained on this data should still be decontaminated against those benchmarks before being scored on them [2][3][5][6]. The MTS-dialog config additionally cannot be loaded at all due to a path defect in the repository's own YAML [2][3][4].

### The screening row

The row's own note [10]: "Medical SFT: MedQA-CoT and MedBooks-18-CoT reasoning written by GPT-4, ChatDoctor cleaned and MedQA-dialog simulated by GPT-3.5, plus human MedMCQA/LiveQA/MTS-dialog. cc-by-nc." Its flag: "contamination risk: same MedQA / MedMCQA lineage, split not stated on card; also cc-by-nc."
