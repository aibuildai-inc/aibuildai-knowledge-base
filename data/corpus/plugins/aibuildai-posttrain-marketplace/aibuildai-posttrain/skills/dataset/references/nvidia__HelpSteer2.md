# nvidia/HelpSteer2

21,362 prompt-response rows, each rated by human annotators on five attributes (helpfulness, correctness, coherence, complexity, verbosity) on a 0-4 scale, split into a 20,324-row train set and a 1,038-row validation set.

**nvidia/HelpSteer2** is a permissively licensed helpfulness-rating dataset built by NVIDIA in partnership with Scale AI, introduced in "HelpSteer2: Open-source dataset for training top-performing reward models" [1] as a follow-up to the earlier HelpSteer release, which the card recommends against using in favor of this one [2]. Each row pairs a prompt with one response and five Likert-scale (0-4) human annotations of that response; consecutive same-prompt row pairs (row 1 with row 2, row 3 with row 4, ...) let the helpfulness scores double as preference pairs, and the same rows support multi-attribute (SteerLM) regression reward-model training [2]. **The `train`/`validation` split is the dataset's own designated train/held-out split - hold out `validation` (1,038 rows) rather than mixing it into training [2].** It lives at https://huggingface.co/datasets/nvidia/HelpSteer2 .

**Use it for**: reward-model training - either multi-attribute regression RM training on the five per-response scores, or Bradley-Terry/DPO-style preference-pair training by taking the helpfulness-score difference between consecutive same-prompt rows, or by loading the dedicated `preference` data_dir instead (see Neighbors) [2]. This maps to the reward-model method card for the regression use, or the DPO method card once pairs are extracted for the preference use. Not a chat-SFT dataset - the `response` field is a single rated reply, not a full assistant turn meant for imitation.

**Licence**: CC-BY-4.0 (`cardData.license` is `"cc-by-4.0"`, tag `license:cc-by-4.0`), ungated (`"gated": false`, `"private": false`) [3]. No further catch found: the README repeats "CC-BY-4.0" as the license for both HelpSteer2 itself and models trained only on it, and states no other restriction [2].

**Shape**: 21,362 rows in one served config (`default`), split `train` 20,324 / `validation` 1,038, seven columns [4][5]. Two further files (`preference/preference.jsonl.gz`, `disagreements/disagreements.jsonl.gz`) sit in the repository tree outside the served config; see Shape below.

**Hold out**: `validation` (1,038 rows) - the dataset's own held-out split, used in the README's own load example as the 5% validation partition [2]. No source states an overlap between HelpSteer2 and any external evaluation benchmark.

**Origin**: built by NVIDIA in partnership with Scale AI; responses come from ten in-house LLMs (not from proprietary providers such as OpenAI) and the five attribute scores are human annotations by roughly 1,000 Scale AI-sourced annotators [2]. Hub API at the check date: `downloads` 28,965, `downloadsAllTime` 512,652, `likes` 454 [3].

**Trained-on-by**: NVIDIA's own Llama-3.1-Nemotron-70B-Reward and Llama-3.1-Nemotron-70B-Instruct models, trained on HelpSteer2 together with the HelpSteer2-Preference annotations, reaching 94.1 on RewardBench [2][6]. The README's own RewardBench table also lists Nemotron-4-340B-Reward and Llama3-70B-SteerLM-RM as reward models trained only on "Permissive Licensed Data Only (CC-BY-4.0)", scoring 92.0 and 88.8 respectively, though it does not state HelpSteer2 by name as their exclusive training set [2].

**Introduced by**: [1] (Wang et al.), with a follow-up preference-annotation release described in [6] (Wang et al.).

## Shape

Rows served and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 20,324 |
| `validation` | 1,038 |
| total | 21,362 |

One served config, `default`, with seven columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `prompt` | string |
| `response` | string |
| `helpfulness` | int64 |
| `correctness` | int64 |
| `coherence` | int64 |
| `complexity` | int64 |
| `verbosity` | int64 |

The repository tree also carries `preference/preference.jsonl.gz` and `disagreements/disagreements.jsonl.gz`, neither of which appears as a config in `/info` - they load only via `load_dataset("nvidia/HelpSteer2", data_dir=...)` [2][5]. Decompressing both directly from the repository: `preference.jsonl.gz` holds 9,125 lines, each carrying a row-level `split` field, of which 8,677 read `"split": "train"` and 448 read `"split": "val"` [7]; the README states that this `split` field is what a user should use to separate the file's train- and val-origin rows, since a plain `load_dataset(..., data_dir="preference")` call returns them all under one `train` key [2]. `disagreements.jsonl.gz` holds 23,652 lines, each with the same `prompt`/`response` schema as `default` but with `helpfulness`/`correctness`/`coherence`/`complexity`/`verbosity` as lists of 2-5 individual annotator scores rather than single averaged integers [2][7]. No source states sequence-length or token statistics for any of the three files.

The card states that about 29% of all prompts are multi-turn, in which case `prompt` holds all user turns and all but the last assistant turn, and `response` holds only the final assistant turn [2]. Sampling the first 96 served `train` rows and the first 100 served `validation` rows (offset 0 of each split, via datasets-server `/first-rows`) found zero multi-turn prompts in either sample, so the 29% figure is taken from the card, not reproduced here [2][8][9].

## Quality

- Every attribute score is a human judgment: contractors sourced through Scale AI rated each response on a 0-4 Likert scale for each of the five attributes, after English-proficiency screening and a 35-sample qualifying test [2].
- Every sample was independently annotated by three to five annotators (mean 3.41), with a fifth annotator added if the first three disagreed by more than 2 points on helpfulness; the final label is the mean of the three most-agreeing annotators, rounded to the nearest integer [2].
- Scale AI performed two-plus human reviews per annotation plus automated checks, after which NVIDIA's own independent QA pass filtered the raw annotations down to the 20,324 retained training samples; the card does not state how many raw samples were filtered out [2].
- The deciding number for whether the dataset "works": a reward model trained on HelpSteer2 alone reached 92.0% on RewardBench's primary dataset, which the origin paper states was the best score among all listed open and proprietary models as of 2024-06-12 [1]. The follow-up HelpSteer2-Preference release, adding the dedicated `preference` annotations on top of the same prompts, pushed a Llama-3.1-70B-Instruct-based reward model to 94.1 on RewardBench, "top of more than 140 reward models as of 1 Oct 2024" [6].
- `disagreements.jsonl.gz` exists specifically so a user can inspect annotator disagreement directly rather than trust only the averaged score; a small share of its rows carry only 2 annotator scores because one or more annotations were marked not-ratable or invalid [2].

## Load it

Train on `train`, hold out `validation`, and pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-12-18) [3]:

```python
import datasets

REV = "990b2711a36180dd19d9c94b8627844866f8982a"  # main at the check date
train = datasets.load_dataset("nvidia/HelpSteer2", revision=REV, split="train")           # 20,324 rows
val = datasets.load_dataset("nvidia/HelpSteer2", revision=REV, split="validation")         # 1,038 rows - hold out

# separate files, outside the default config, loaded by data_dir:
preference = datasets.load_dataset("nvidia/HelpSteer2", revision=REV, data_dir="preference")["train"]     # 9,125 rows total, both train and val prompts - use its own "split" column to separate them
disagreements = datasets.load_dataset("nvidia/HelpSteer2", revision=REV, data_dir="disagreements")["train"]  # 23,652 rows
```

**Trap**: `load_dataset("nvidia/HelpSteer2", data_dir="preference")` returns everything under a single `train` split key regardless of the original train/validation origin of each row - the README warns "despite the name, this contains both train and val, which you can use split to distinguish" [2]; a reader who trains on this `train` key without filtering the row-level `split` field will leak validation-origin prompts into training.

## Neighbors

- `nvidia/HelpSteer` - the direct predecessor, 37,120 rows (35,331 train / 1,789 validation) [10]. Its own card's declared schema names the same seven columns as HelpSteer2 (`prompt`, `response`, `helpfulness`, `correctness`, `coherence`, `complexity`, `verbosity`), though it types the five score columns as `int32` where HelpSteer2 serves them as `int64` [11]. The HelpSteer2 card itself says "we recommend using HelpSteer2 instead of HelpSteer" [2].
- `nvidia/HelpSteer3` - the successor, a five-config release (`preference` 40,476 rows, `edit` 14,461, `edit_quality` 3,274, `feedback` 40,821, `principle` 33,905) spanning 14 languages rather than English only; its own card says its preference data trains Llama 3.3 Nemotron models [12][13].
- `RLHFlow/Helpsteer2-standard` - a third-party binarized reformatting of this dataset's ratings into `chosen`/`rejected` message lists with `chosen_score`/`rejected_score` floats, 8,053 rows in a single `train` split - roughly the size of the `preference` data_dir above but reshaped into a DPO-ready chat format rather than the paired-row or explicit-preference-annotation forms this repository ships [12].
- This corpus prefers the base `nvidia/HelpSteer2` repository over the binarized/reformatted community forks: the reformattings fix one training shape (e.g. DPO message pairs), while the base repository keeps the full five-attribute ratings, the dedicated preference-strength annotations, and the raw disagreement data all in one place, letting a chooser pick the shape at load time instead of only what a fork already committed to.

## A row

The repository serves one config with two splits sharing the same schema, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8]:

```json
{
  "prompt": "c#",
  "response": "C# is a high-level, object-oriented programming language developed by Microsoft as part of its .NET initiative. It was created as a modern alternative to Java and supports a variety of programming paradigms, including imperative, functional, and event-driven. C# is primarily used for Windows application development, but it can also be used for web, mobile, and game development. The language is designed to be safe, secure, and efficient, and it provides developers with a rich set of libraries and tools for building robust and scalable applications. C# is also widely used in the game development industry, particularly in the development of games for the Xbox 360 and Xbox One consoles.",
  "helpfulness": 3,
  "correctness": 4,
  "coherence": 4,
  "complexity": 2,
  "verbosity": 1
}
```

The `validation` split shares the identical schema; its first served row, `config="default"`, `split="validation"`, `row_idx=0` [9]:

```json
{
  "prompt": "explain master slave replication nsql",
  "response": "In the context of NoSQL databases, master-slave replication refers to a configuration where a single master node writes data, and one or more slave nodes read data from the master and replicate it to provide read scalability. The master node is responsible for accepting write requests and updating i...",
  "helpfulness": 4,
  "correctness": 4,
  "coherence": 4,
  "complexity": 3,
  "verbosity": 2
}
```

The two tree-only files carry different schemas, read directly from the decompressed repository files rather than through datasets-server, which does not index them [7]. One row of `preference/preference.jsonl.gz` (truncated justification fields):

```json
{
  "split": "train",
  "prompt": "Define Signal Discuss its various properties with the help of diagram",
  "response_1": "A signal is a form of energy that is used to transmit information from one place to another...",
  "response_2": "A signal is a message that is conveyed from a sender to a receiver through a communication channel...",
  "preference_strength": 1,
  "preference_statement": "@Response 2 is better than @Response 1 because it provides a comprehensive insightful explanation of signanal and its properties.",
  "preference_elaboration": "It is complete, clear and correct as it discuss all the the poperties of signal while @Response 1 only discusses three properties of signal. ..."
}
```

One row of `disagreements/disagreements.jsonl.gz`, the same first prompt/response as the `default`-config example above but with per-annotator score lists instead of averaged integers:

```json
{
  "prompt": "c#",
  "response": "C# is a high-level, object-oriented programming language developed by Microsoft as part of its .NET initiative...",
  "helpfulness": [3, 3, 4],
  "correctness": [3, 4, 4],
  "coherence": [4, 3, 4],
  "complexity": [2, 2, 2],
  "verbosity": [2, 1, 1]
}
```

## Where it came from

Built by NVIDIA in partnership with Scale AI [2]. Prompts are mostly user-contributed ShareGPT prompts, with about 5% written directly by Scale AI [2]. Responses are generated by early versions of ten different in-house NVIDIA LLMs - explicitly none from proprietary providers such as OpenAI - with two responses drawn per prompt from two different models, using sampling to diversify outputs [2]. Attribute annotation was done by roughly 1,000 U.S.-based Scale AI-sourced contractors under the process described in Quality above, and Scale AI states it paid annotators via the Anker Methodology and GISC Impact Sourcing Standard [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. datasets-server endpoints take no revision parameter and reflect the live repository state at fetch time.

[1] Wang et al., "HelpSteer2: Open-source dataset for training top-performing reward models", 2024. https://arxiv.org/abs/2406.08673 - the origin paper; current title read from the live abs page; RewardBench 92.0% SOTA claim as of 2024-06-12. Fetched 2026-08-11.

[2] nvidia/HelpSteer2 dataset card (README). https://huggingface.co/datasets/nvidia/HelpSteer2/raw/main/README.md - dataset description, attribute definitions, preference/disagreements file descriptions, load instructions, source and annotation methodology, ethical statement. Fetched 2026-08-11.

[3] Hugging Face Hub API record for nvidia/HelpSteer2. https://huggingface.co/api/datasets/nvidia/HelpSteer2?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=nvidia%2FHelpSteer2 Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=nvidia%2FHelpSteer2 Fetched 2026-08-11.

[6] Wang et al., "HelpSteer2-Preference: Complementing Ratings with Preferences", 2024. https://arxiv.org/abs/2410.01257 - follow-up preference-annotation paper; current title read from the live abs page; 94.1 RewardBench claim. Fetched 2026-08-11.

[7] `preference/preference.jsonl.gz` and `disagreements/disagreements.jsonl.gz`, fetched directly from the repository (`resolve/main`) and decompressed locally to count lines and read schema/rows - these two files are not indexed by datasets-server. https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/preference/preference.jsonl.gz and https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/disagreements/disagreements.jsonl.gz Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, train split. https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FHelpSteer2&config=default&split=train Fetched 2026-08-11.

[9] datasets-server first-rows endpoint, validation split. https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FHelpSteer2&config=default&split=validation Fetched 2026-08-11.

[10] datasets-server size endpoint for nvidia/HelpSteer. https://datasets-server.huggingface.co/size?dataset=nvidia%2FHelpSteer - live, not pinned. Fetched 2026-08-11.

[11] nvidia/HelpSteer dataset card (README). https://huggingface.co/datasets/nvidia/HelpSteer/raw/main/README.md - declared `dataset_info.features` schema (column names and dtypes). Fetched 2026-08-11.

[12] datasets-server size endpoint, one call per neighbor: nvidia/HelpSteer3, RLHFlow/Helpsteer2-standard. https://datasets-server.huggingface.co/size?dataset=<id> - live, not pinned. Fetched 2026-08-11.

[13] nvidia/HelpSteer3 dataset card (README). https://huggingface.co/datasets/nvidia/HelpSteer3/raw/main/README.md - config list, language list, stated use for Llama 3.3 Nemotron training. Fetched 2026-08-11.

[14] The corpus screening row for `nvidia/HelpSteer2`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reward-model training data: train on `train`, hold out `validation`. The card's own scope is a rating/reward-model dataset, not a chat-SFT dataset, and its own README recommends using HelpSteer2 rather than HelpSteer for this purpose - both facts are already established above [2]. The screening row's note names the same collection process this card documents in Where it came from [14].

### The screening row

The row's own note [14]: "human helpfulness attributes collected with Scale AI over responses from several models." The row carries no flag.
