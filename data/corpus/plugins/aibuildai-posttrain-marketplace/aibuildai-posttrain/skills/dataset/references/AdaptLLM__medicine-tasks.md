# AdaptLLM/medicine-tasks

5,383 pre-templatized zero/few-shot test prompts across five biomedical benchmarks - ChemProt, MQP, PubMedQA, RCT, and USMLE - each row a filled-in prompt with a candidate-answer list and a gold index, released as evaluation data only.

**AdaptLLM/medicine-tasks** packages the biomedicine test sets used in "Adapting Large Language Models to Domains via Reading Comprehension" [1], the paper behind the AdaptLLM domain-adaptation method. The repository's own card describes it as the "**evaluation datasets**" for that paper, releasing "the filled-in zero/few-shot input instructions and output completions" so the paper's prompting results can be reproduced on any Hugging Face model [2]. The README names the five configs only as `ChemProt`, `MQP`, `PubMedQA`, `RCT`, and `USMLE`, without describing what each task tests [2]; the row sampled from each config's `test` split (below) shows the task shape directly - `ChemProt` queries a chemical-protein relation label from a labeled entity pair, `MQP` asks whether two medical questions are paraphrases, `PubMedQA` asks a yes/no/maybe question over a PubMed abstract, `RCT` asks for a sentence's structural role in an abstract, and `USMLE` is a 4-option medical exam question [3]. **This is evaluation data, not training data: every split is `test`, and the README frames the whole repository as reproduction material for prompting results, never as a fine-tuning corpus [2]. Hold out all 5,383 rows from any training set.** It lives at https://huggingface.co/datasets/AdaptLLM/medicine-tasks .

**Use it for**: nothing to train on - this is a held-out zero/few-shot evaluation benchmark for scoring a model's biomedical prompting ability, not a source of preference, SFT, or reasoning-trace training examples [2]. It does not map to any training-format method card.

**Licence**: not stated - the Hub record's `license` field is empty, the YAML card metadata carries no `license:` key, and no licence is named anywhere in the card body [4]. The repository is ungated [4].

**Shape**: 5 configs (`ChemProt`, `MQP`, `PubMedQA`, `RCT`, `USMLE`), one `test` split each, 5,383 rows total, 4 columns per config [5][6].

**Hold out**: all 5,383 rows, across every config - the whole repository is test data for the five named benchmarks and none of it should enter a training set [2]. The corpus screening note makes the same call [7].

**Origin**: released by the AdaptLLM Hugging Face organization; the config names (`ChemProt`, `MQP`, `PubMedQA`, `RCT`, `USMLE`) and the sampled rows show these are pre-existing biomedical benchmark items reformatted into few-shot prompts, not model-generated text [2][3]. Hub API at the check date: `downloads` 1,755, `downloadsAllTime` 57,638, `likes` 33 [4].

**Trained-on-by**: none found. The README explicitly separates this repository (evaluation prompts) from a different set of "raw training and testing splits...for facilitating fine-tuning" that it hosts under other repository names [2], so no source documents a model training on this repository's rows.

**Introduced by**: [1] (Cheng, Huang, and Wei; ICLR 2024).

## Shape

Configs and rows (datasets-server `/size`) [5]:

| config | split | rows |
| --- | --- | --- |
| `ChemProt` | `test` | 500 |
| `MQP` | `test` | 610 |
| `PubMedQA` | `test` | 1,000 |
| `RCT` | `test` | 2,000 |
| `USMLE` | `test` | 1,273 |
| total | | 5,383 |

Every config shares one 4-column schema (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `input` | string |
| `options` | list\<string\> |
| `gold_index` | int64 |

`input` holds a full zero/few-shot prompt (prior demonstrations plus the final, unanswered query); `options` is the list of candidate answers for that query; `gold_index` is the index into `options` that is correct. No source states token or sequence-length statistics for these prompts; not stated.

Sizes (datasets-server `/size`) [5]: 11,765,386 bytes of original JSON, 5,133,517 bytes as Parquet, 10,881,220 bytes decoded in memory.

`/size` and `/info` take no revision parameter and are live; their own `download_checksums` point at commit `6dcbdb09dac1f8e7328178686cfcdea93ae78d45` (2024-08-29) rather than the `68afee8d...` sha pinned in Load it (2024-12-02, the shortlist row's `commit`) [8]. The commit history between the two shows two intervening commits, both titled "Update README.md" [8]; comparing the two commits' recursive file trees directly shows this reconciles cleanly - the five `test.json` blob hashes are identical at both commits, and only `README.md`'s blob differs [8]. So the row counts, sizes, and columns above hold at the pinned revision too, even though the datasets-server responses above were generated against the earlier commit.

## Quality

- No source states a measured error rate, duplicate rate, or annotator-agreement figure for this repository; none is invented here.
- The five sampled row-0 prompts (one per config, read live) [3] all end immediately after the final query with no answer text appended - the gold answer lives only in `gold_index`/`options`, not concatenated into `input` - which is the structural property that keeps the completion out of the prompt a model is scored on.
- The README does not restate error rates or known issues for any of the five tasks, and none are invented here; the row-0 sample for `ChemProt` shows the row's own relation vocabulary runs to 13 labels (`product-of`, `substrate_product-of`, `indirect-downregulator`, `activator`, `indirect-upregulator`, `substrate`, `upregulator`, `inhibitor`, `agonist-activator`, `downregulator`, `agonist`, `antagonist`, `agonist-inhibitor`) [3].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date, matching the shortlist row's `commit`) [4], and load one config at a time - `load_dataset` requires a config name here, there is no default:

```python
import datasets

REV = "68afee8d7be55335ba0e34deacc23120233161f6"  # main at the check date
chemprot = datasets.load_dataset("AdaptLLM/medicine-tasks", "ChemProt", revision=REV, split="test")   # 500 rows
mqp      = datasets.load_dataset("AdaptLLM/medicine-tasks", "MQP", revision=REV, split="test")         # 610 rows
pubmedqa = datasets.load_dataset("AdaptLLM/medicine-tasks", "PubMedQA", revision=REV, split="test")    # 1,000 rows
rct      = datasets.load_dataset("AdaptLLM/medicine-tasks", "RCT", revision=REV, split="test")         # 2,000 rows
usmle    = datasets.load_dataset("AdaptLLM/medicine-tasks", "USMLE", revision=REV, split="test")       # 1,273 rows
```

**Trap**: the config names in the YAML metadata do not match the directory names in the repository tree - `MQP` reads from `MedQs/test.json` and `PubMedQA` reads from `pubmed_qa/test.json`, both lower/differently-cased directories that do not equal the config string [2][9]. Passing `"MedQs"` or `"pubmed_qa"` as the config name to `load_dataset` fails; only the five YAML-declared names (`ChemProt`, `MQP`, `PubMedQA`, `RCT`, `USMLE`) work.

## Neighbors

- `AdaptLLM/ChemProt` - the same paper's raw, non-templated ChemProt data, with `train` (4,169 rows), `validation` (2,427), and full `test` (3,469) splits over plain `text`/`label` columns, versus the 500-row templated `test` subset here [10][2]. A row-0 comparison shows the two repositories store the same underlying sentence-classification task in different shapes: raw `text`/`label` pairs there against a few-shot `input`/`options`/`gold_index` prompt here [11]. This card's `ChemProt` row 0 ends on the query sentence "Moreover, geldanamycin decreases the amount and phosphorylation of Lck and Raf-1 kinases..." with gold label `indirect-downregulator`; searching `AdaptLLM/ChemProt`'s raw `test` split for that sentence returns 363 keyword matches, of which the first 100 (the page actually read) include eight rows sharing the same text with different entity spans - three (row_idx 142, 143, 144) carrying the matching label `INDIRECT-DOWNREGULATOR` and five carrying `INHIBITOR` for a different entity pair in the same sentence; the remaining 263 keyword matches were not read [12]. That one-row check confirms this config's rows are drawn from the same raw ChemProt test pool - it does not establish that all 500 `ChemProt` rows here overlap `AdaptLLM/ChemProt`'s `test` split, only that at least this one does; treat the two repositories' `ChemProt`/`test` data as the same held-out region rather than verify-and-reuse it as additive data.
- `AdaptLLM/RCT` - likewise the raw RCT data behind this repository's RCT config, with `train` (180,040), `validation` (30,212), and full `test` (30,135) splits, versus the 2,000-row templated `test` subset here [13]. This card's `RCT` row 0 ends on the query sentence "Among medical patients in the high exposure arm, the use of ciprofloxacin and piperacillin/tazobactam was 51% and 75% higher..." with gold label `Results`; a keyword search for that sentence in `AdaptLLM/RCT`'s raw `test` split returns 7,747 matches; among the first 100 (the page actually read), exactly one row is an exact-text match, row_idx 15595, labeled `RESULTS` - the other 7,647 keyword matches and the rest of the 30,135-row split were not read [14]. As with `ChemProt`, this is a single verified row, not a check of the other 1,999 rows - but it is enough to treat `AdaptLLM/RCT`'s `test` split as the same held-out region as this repository's RCT rows, not separate additive data.
- The README lists raw train/test repositories for other domains and tasks (`ConvFinQA`, `FiQA_SA`, `Headline`, `NER`, `FPB`) but names no raw repository for MQP, PubMedQA, or USMLE in the medicine domain - so for those three tasks, no sibling raw-split repository was found [2].
- `AdaptLLM/finance-tasks` and `AdaptLLM/law-tasks` are the same paper's evaluation packages for the other two domains it studies; they cover different tasks entirely and carry no medicine-domain overlap [2].
- `AdaptLLM/med_knowledge_prob` is a separate knowledge-probing dataset from the same organization, not an evaluation-task package, and is not a duplicate of this repository's rows [2].

## A row

All five configs share one schema (`id`, `input`, `options`, `gold_index`); one row from each is shown to make the shared shape and the per-task prompt content both checkable. Read live, row 0 of `test` in each config (datasets-server `/first-rows`) [3], long `input` fields truncated with an ellipsis marker:

```json
// config="USMLE", split="test", row_idx=0
{
  "id": 0,
  "input": "A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. [...] When phenol is applied to a sample of the patient's blood at 90°C, a phosphorylated N-acetylglucosamine dimer with 6 fatty acids attached to a polysaccharide side chain is identified. A blood culture is most likely to show which of the following?",
  "options": [
    "Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar",
    "Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal blood agar",
    "Spore-forming, gram-positive bacilli forming yellow colonies on casein agar",
    "Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar"
  ],
  "gold_index": 3
}
```

```json
// config="PubMedQA", split="test", row_idx=0
{
  "id": 0,
  "input": "Article: (Objective) We evaluated the usefulness of a short stay or 23-hour ward in a pediatric unit [...] Now answer this question: A short stay or 23-hour ward in a general and academic children's hospital: are they effective?",
  "options": ["No", "Maybe", "Yes"],
  "gold_index": 2
}
```

The other three configs (`ChemProt`, `MQP`, `RCT`) follow the identical `id`/`input`/`options`/`gold_index` shape with task-specific prompt text and option lists (13 relation labels for `ChemProt`, `["No", "Yes"]` for `MQP`, five section-role labels for `RCT`) [3].

## Where it came from

Released by the AdaptLLM Hugging Face organization as the reproduction material for the biomedicine evaluations in "Adapting Large Language Models to Domains via Reading Comprehension" [1]. The README states the repository holds "the filled-in zero/few-shot input instructions and output completions of the test" for each domain-specific task, explicitly noting these prompts "are specifically tailored for models before alignment and do NOT fit for the specific data format required for chat models" [2]. It separately points to raw, non-templated train/test splits hosted under per-task repository names (including `AdaptLLM/ChemProt` and `AdaptLLM/RCT` for medicine) "for facilitating fine-tuning or other usages" [2] - a different, training-oriented artifact from the eval-only prompts in this repository.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. The corpus screening row lives in the pipeline's own shortlist data supplied with this task, cited by dataset id, never by line or entry number.

[1] Cheng, Huang, and Wei, "Adapting Large Language Models to Domains via Reading Comprehension", ICLR 2024. https://arxiv.org/abs/2309.09530 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] AdaptLLM/medicine-tasks dataset card (README). https://huggingface.co/datasets/AdaptLLM/medicine-tasks/raw/main/README.md - repository description, task list, usage notes, raw-dataset pointers, config YAML. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint, one call per config. https://datasets-server.huggingface.co/first-rows?dataset=AdaptLLM%2Fmedicine-tasks&config=<ChemProt|MQP|PubMedQA|RCT|USMLE>&split=test Fetched 2026-08-11.

[4] Hugging Face Hub API record for AdaptLLM/medicine-tasks. https://huggingface.co/api/datasets/AdaptLLM/medicine-tasks?full=true - licence, gate, `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=AdaptLLM%2Fmedicine-tasks Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=AdaptLLM%2Fmedicine-tasks Fetched 2026-08-11.

[7] The corpus screening row for `AdaptLLM/medicine-tasks`, supplied with this card's request - its `note`, read back in the appendix. Checked 2026-08-11.

[8] Hugging Face Hub commit-history and recursive tree-listing APIs for AdaptLLM/medicine-tasks, fetched at both the pinned commit and the commit datasets-server's `/info` checksums reference. https://huggingface.co/api/datasets/AdaptLLM/medicine-tasks/commits/main and https://huggingface.co/api/datasets/AdaptLLM/medicine-tasks/tree/<sha>?recursive=true (for `sha` = `68afee8d7be55335ba0e34deacc23120233161f6` and `6dcbdb09dac1f8e7328178686cfcdea93ae78d45`) - used to compare blob hashes across the two commits. Fetched 2026-08-11.

[9] Hugging Face Hub tree API for AdaptLLM/medicine-tasks. https://huggingface.co/api/datasets/AdaptLLM/medicine-tasks/tree/main - repository directory names (`ChemProt`, `MedQs`, `RCT`, `pubmed_qa`, `usmle`). Fetched 2026-08-11.

[10] datasets-server size endpoint for the neighbor repository. https://datasets-server.huggingface.co/size?dataset=AdaptLLM%2FChemProt Fetched 2026-08-11.

[11] datasets-server first-rows endpoint for the neighbor repository. https://datasets-server.huggingface.co/first-rows?dataset=AdaptLLM%2FChemProt&config=ChemProt&split=test Fetched 2026-08-11.

[12] datasets-server search endpoint for the neighbor repository, querying the exact sentence from this card's `ChemProt` row 0. https://datasets-server.huggingface.co/search?dataset=AdaptLLM%2FChemProt&config=ChemProt&split=test&query=geldanamycin+decreases+the+amount+and+phosphorylation+of+Lck Fetched 2026-08-11.

[13] datasets-server size endpoint for the neighbor repository. https://datasets-server.huggingface.co/size?dataset=AdaptLLM%2FRCT Fetched 2026-08-11.

[14] datasets-server search endpoint for the neighbor repository, querying the exact sentence from this card's `RCT` row 0. https://datasets-server.huggingface.co/search?dataset=AdaptLLM%2FRCT&config=RCT&split=test&query=Among+medical+patients+in+the+high+exposure+arm Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Eval-only, correctly labeled: the repository is the AdaptLLM paper's biomedical evaluation data, entirely `test` splits, and the card above resolves it to a full 5,383-row hold-out with no training use. This rests on the README's own framing of the repository as reproduction material for prompting results [2] and on the screening row's note [7].

### The screening row

The row's own note [7]: "the AdaptLLM paper's biomedical EVALUATION sets (ChemProt, MQP, PubMedQA, RCT, USMLE); test split only." The row carries no flag.
