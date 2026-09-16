# shibing624/medical

A Chinese-medical training bundle of about 2.44 million rows split across three loading-script configs - a continual-pretraining text corpus, an SFT instruction set, and a preference/reward set - not a single flat table.

**shibing624/medical** is a Chinese medical dataset released by the shibing624 (Xu Ming) Hub account for training medical-domain language models, tied to the author's MedicalGPT and textgen training repositories [1]; there is no separate origin paper - the dataset card itself introduces it [1]. It lives at https://huggingface.co/datasets/shibing624/medical . It ships no Parquet data, only a `medical.py` `GeneratorBasedBuilder` loading script that defines three `BuilderConfig`s - `pretrain`, `finetune`, `reward` - each with its own column schema and its own train/validation/test split [2]. **`load_dataset` requires picking one of the three configs by name (the script sets no default, so calling it with none raises a missing-config-name error); the `pretrain` config's own split generators never read `pretrain/medical_book_zh.json` (8,475 rows), so that file is present in the repo but absent from every served split unless fetched directly; and because both the `pretrain` config's `train_encyclopedia.json` and part of the `finetune` config's Chinese train file independently draw from the same upstream FreedomIntelligence/huatuo_encyclopedia_qa pool, overlap between `pretrain`'s train rows and `finetune`'s held-out test rows is not ruled out by any source (see Hold out and Quality)** [1][2][9].

**Use it for**: three distinct training shapes, chosen by config name - `pretrain` yields single-`text` rows for continual/domain pretraining; `finetune` yields `instruction`/`input`/`output` triples for SFT chat data, mapping to the SFT method card; `reward` yields `question`/`response_chosen`/`response_rejected` triples, an explicit-prompt preference shape (rename to `prompt`/`chosen`/`rejected` for a DPO-style trainer), mapping to the DPO or reward-model method card [2]. Apache-2.0 covers the release itself, not necessarily every upstream source it merges (see Licence).

**Licence**: apache-2.0 (`cardData.license` is `"apache-2.0"`; tags include `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [3]. The one catch: this apache-2.0 tag covers shibing624's release of the merged files, and no source states that the upstream corpora it draws from - Toyhom/Chinese-medical-dialogue-data, Kent0n-Li/ChatDoctor, FreedomIntelligence/huatuo_encyclopedia_qa, FreedomIntelligence/huatuo_knowledge_graph_qa, the MedQA textbook extract, and SCIR-HI/Huatuo-Llama-Med-Chinese's generated replies - were themselves released under apache-2.0 or that this tag supersedes their own terms [1].

**Shape**: 3 configs (`pretrain`, `finetune`, `reward`), each with `train`/`validation`/`test`; served-split totals are 362,420 / 2,068,589 / 4,000 rows respectively (2,435,009 rows total, plus the 8,475 unserved book rows), full per-file counts in Shape below [1][2].

**Hold out**: hold out each config's own `validation`/`test` split before training on its `train` split - `pretrain` valid 500 / test 500, `finetune` valid 1,000 / test 1,000 (Chinese+English combined), `reward` valid 100 / test 100 [1]. Additionally, treat `pretrain/train_encyclopedia.json` and any `finetune` Chinese test rows with caution: both trace independently to the same 364,420-row FreedomIntelligence/huatuo_encyclopedia_qa pool, and no source states the two configs' splits were coordinated to keep them disjoint (see Quality) [1][9].

**Origin**: released by the shibing624 Hub account; content is a merge of human doctor-patient dialogue, encyclopedia/knowledge-graph QA, a medical textbook extract, and one model's generated replies (see Where it came from) [1]. Hub API at the check date (2026-08-11): `downloads` 2,326, `downloadsAllTime` 41,427, `likes` 439 [3].

**Trained-on-by**: 18 Hub models declare `dataset:shibing624/medical` in their tags as of the check date, including `sethuiyer/Dr_Samantha-7b` (46 downloads, 23 likes) and its GGUF/AWQ/GPTQ requantizations by TheBloke, mradermacher and tensorblock, plus `Sirius27/BeingWell_llama2_7b`, `NewstaR/StableGalen-6b`, and two DeepSeek-R1-Distill-Qwen medical fine-tunes by `beita6969` [7].

**Introduced by**: no paper - the dataset card [1].

## Shape

Per-file row counts, from the card's own `wc -l medical/*/*` listing [1], with byte sizes read from the Hub file tree at the pinned commit [4]:

| config | file | rows | bytes |
| --- | --- | --- | --- |
| pretrain | `train_encyclopedia.json` | 361,420 | 591,029,894 |
| pretrain | `valid_encyclopedia.json` | 500 | 805,308 |
| pretrain | `test_encyclopedia.json` | 500 | 817,965 |
| pretrain | `medical_book_zh.json` (not loaded by any split) | 8,475 | 40,157,289 |
| finetune | `train_zh_0.json` | 1,949,972 | 1,338,359,048 |
| finetune | `train_en_1.json` | 116,617 | 139,103,795 |
| finetune | `valid_zh_0.json` | 500 | 306,784 |
| finetune | `valid_en_1.json` | 500 | 609,216 |
| finetune | `test_zh_0.json` | 500 | 298,414 |
| finetune | `test_en_1.json` | 500 | 602,007 |
| reward | `train.json` | 3,800 | 3,088,737 |
| reward | `valid.json` | 100 | 81,876 |
| reward | `test.json` | 100 | 107,406 |

The loading script concatenates files per config: `pretrain`'s `train` split is `train_encyclopedia.json` alone (`medical_book_zh.json` is never referenced in `_split_generators`); `finetune`'s `train` split merges `train_zh_0.json` and `train_en_1.json` into one 2,066,589-row split (and likewise `valid`/`test` merge their zh/en halves into 1,000/1,000-row splits); `reward`'s splits are each a single file [2]. Columns are `text` (pretrain); `instruction`, `input`, `output` (finetune); `question`, `response_chosen`, `response_rejected` (reward), all `string`-typed [2]. No source states sequence-length or token statistics for any config; none is invented here.

## Quality

- The `reward/train.json` prose says the questions are "共4000条" (4,000) sampled from Toyhom/Chinese-medical-dialogue-data, but the same card's own `wc -l` table measures 3,800 lines in that file - a discrepancy the card does not resolve; the measured count (3,800) is used everywhere else on this card because it is checked against the file's own byte size (3,088,737 bytes) [1].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset; none is invented here.
- The `reward` config's dispreference signal is not a human judgment of quality: `response_chosen` is the original doctor's reply from Toyhom/Chinese-medical-dialogue-data, while `response_rejected` is a reply generated by the SCIR-HI/Huatuo-Llama-Med-Chinese model - so the pair records human-reply-vs-one-model-reply, not a graded human preference between two candidates [1].
- `pretrain/train_encyclopedia.json` (361,420 rows) and part of `finetune/train_zh_0.json` (about 360,000 of its 1,949,972 rows, by the card's own count) both draw from the same upstream FreedomIntelligence/huatuo_encyclopedia_qa pool, which independently serves 364,420 rows split 362,420/1,000/1,000 [1][9]. No source states how - or whether - the `pretrain` and `finetune` configs' own train/valid/test splits were coordinated against that shared pool, so a `finetune` Chinese eval row and a `pretrain` train row are not guaranteed distinct.

## Load it

```python
import datasets

REV = "6e219f1a14856833ee436063d3b73c5f1ab9cfb9"  # main at the check date

pretrain = datasets.load_dataset("shibing624/medical", "pretrain", revision=REV, trust_remote_code=True)
# train 361,420 / validation 500 / test 500 - "text" only; medical_book_zh.json (8,475 rows) is NOT included

finetune = datasets.load_dataset("shibing624/medical", "finetune", revision=REV, trust_remote_code=True)
# train 2,066,589 (zh+en merged) / validation 1,000 / test 1,000 - "instruction"/"input"/"output"

reward = datasets.load_dataset("shibing624/medical", "reward", revision=REV, trust_remote_code=True)
# train 3,800 / validation 100 / test 100 - "question"/"response_chosen"/"response_rejected"
```

**Traps**: (1) a config name is mandatory - `datasets.load_dataset("shibing624/medical")` with no second argument fails, because `medical.py` defines three `BuilderConfig`s and sets no `default_config_name` [2]. (2) this repository ships a `.py` loading script rather than Parquet/Arrow data, so recent `datasets` versions need `trust_remote_code=True`, and it is also why the Hub's dataset viewer and the datasets-server `/info` and `/size` endpoints return HTTP 501 ("doesn't support this dataset because it runs arbitrary Python code") for this repo instead of a live schema [2][5]. (3) loading `pretrain` never yields `medical_book_zh.json`'s 8,475 textbook rows - to get them, download the file directly (e.g. `hf_hub_download(repo_id="shibing624/medical", filename="pretrain/medical_book_zh.json", repo_type="dataset", revision=REV)`) and parse its JSON-lines yourself [2].

## Neighbors

- FreedomIntelligence/huatuo_encyclopedia_qa - the direct upstream for `pretrain`'s `train_encyclopedia.json`/`valid_encyclopedia.json`/`test_encyclopedia.json` and for part of `finetune`'s `train_zh_0.json`; served live at 364,420 rows (362,420 train / 1,000 validation / 1,000 test), close to but not identical to this release's pretrain-config counts (361,420/500/500) [9][1]. If a run already trains on this upstream directly, skip `shibing624/medical`'s `pretrain` config to avoid duplicating it.
- shibing624/huatuo_medical_qa_sharegpt - same builder, a separate Chinese medical SFT set converted to ShareGPT `conversations` format from FreedomIntelligence's HuatuoGPT-sft-data-v1 (226,042 rows) and HuatuoGPT2_sft_instruct_GPT4_50K (50,000 rows) [10]; its upstream and column schema do not overlap with `shibing624/medical`'s `finetune` config (`instruction`/`input`/`output`), so it is a sibling release, not a duplicate.
- A Hub dataset search for the file names unique to this repo (`medical_book_zh`, `huatuo_encyclopedia_qa`) found no Parquet-converted mirror or cleaned/binarized re-release of `shibing624/medical` itself [11].

## A row

Three distinct served shapes, one row each, fetched live via byte-range GET on the raw file (long fields truncated with `[...]`) [6]:

`config="pretrain"`, `split="train"`, from `pretrain/train_encyclopedia.json`:

```json
{"text": "毒蕈中毒预防是什么？1.切勿采摘自己不认识的蘑菇食用。2.毫无识别毒蕈经验者，千万不要自采蘑菇。3.预防措施：加强宣传、避免误食。4.有毒野生菇（菌）类常具备以下特征：[...]"}
```

`config="finetune"`, `split="train"`, from `finetune/train_zh_0.json` (the merged Chinese+English `train` split also contains English rows shaped like `finetune/train_en_1.json`, same three columns):

```json
{"instruction": "血热的临床表现是什么?", "input": "", "output": "初发或复发病不久。皮疹发展迅速，呈点滴状、钱币状或混合状。常见丘疹、斑丘疹、大小不等的斑片，潮红、鲜红或深红色。[...]"}
```

`config="reward"`, `split="train"`, from `reward/train.json`:

```json
{"question": "治疗阳痿吃什么药呢？，性生活一直很正常的，但是这段时间感觉性欲变低了，有时勃起都感觉很困难，[...]", "response_chosen": "男子早泄、早泄病症的再次发生，多由恣情纵欲，或青年误犯性交，至命门火衰，精气虚寒；[...]", "response_rejected": "建议家长先带孩子去正规医院做全面检查以确定病因和病情严重程度；同时可以进行物理治疗、康复训练等辅助治疗方法。"}
```

## Where it came from

Released by the shibing624 Hub account, tied to the MedicalGPT and textgen training repositories linked from the card [1]. Each config merges different upstream pools [1]:

- `pretrain/train_encyclopedia.json` (361,420 rows): question and answer text from FreedomIntelligence/huatuo_encyclopedia_qa concatenated into one `text` field.
- `pretrain/medical_book_zh.json` (8,475 rows, not loaded by any split): passages cut to about 2,048 characters from a Chinese medical-textbook corpus originally hosted for the MedQA project (jind11/MedQA on GitHub).
- `finetune/train_zh_0.json` (1,949,972 rows): merges about 790,000 rows of six-department doctor-patient dialogue from Toyhom/Chinese-medical-dialogue-data, about 360,000 rows from huatuo_encyclopedia_qa, and about 790,000 rows from FreedomIntelligence/huatuo_knowledge_graph_qa.
- `finetune/train_en_1.json` (116,617 rows): English doctor-patient dialogue merging the HealthCareMagic-100k and GenMedGPT-5k subsets of Kent0n-Li/ChatDoctor.
- `reward/train.json`: questions sampled from Toyhom/Chinese-medical-dialogue-data; `response_chosen` is that dataset's original doctor reply; `response_rejected` is a reply generated by the SCIR-HI/Huatuo-Llama-Med-Chinese model.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hugging Face Hub repositories are mutable (a repo can be force-pushed or its README edited), which is why Load it pins the revision; the datasets-server endpoints in [5] take no revision parameter and reflect the live repo state at fetch time, not the pinned commit.

[1] shibing624/medical dataset card (README). https://huggingface.co/datasets/shibing624/medical/raw/main/README.md - description, provenance of every file, the `wc -l` row-count table, field definitions, licensing statement, linked upstream/homepage repositories. Fetched 2026-08-11.

[2] shibing624/medical loading script. https://huggingface.co/datasets/shibing624/medical/raw/main/medical.py - the three `BuilderConfig`s, their column schemas, and their `_split_generators`/`_generate_examples` logic (including that `medical_book_zh.json` is never referenced). Fetched 2026-08-11.

[3] Hugging Face Hub API record for shibing624/medical. https://huggingface.co/api/datasets/shibing624/medical?full=true and the same endpoint with `expand[]=downloadsAllTime` - `sha`, `cardData.license`, `gated`, `private`, `downloads`, `downloadsAllTime`, `likes`, `lastModified`, `siblings`. Fetched 2026-08-11.

[4] Hugging Face Hub tree API, recursive with sizes. https://huggingface.co/api/datasets/shibing624/medical/tree/main?recursive=true&expand=true - per-file byte sizes at the pinned commit. Fetched 2026-08-11.

[5] datasets-server info and size endpoints. https://datasets-server.huggingface.co/info?dataset=shibing624%2Fmedical and https://datasets-server.huggingface.co/size?dataset=shibing624%2Fmedical - both return HTTP 501 with the message that the dataset "doesn't support this dataset because it runs arbitrary Python code". Fetched 2026-08-11.

[6] Raw file content fetched by byte-range GET (`curl -r 0-3000`) directly from `https://huggingface.co/datasets/shibing624/medical/resolve/main/<path>` for `pretrain/train_encyclopedia.json`, `pretrain/medical_book_zh.json`, `finetune/train_zh_0.json`, `finetune/train_en_1.json`, and `reward/train.json` - the served rows in A row. Fetched 2026-08-11.

[7] Hugging Face Hub models API, filtered on the dataset tag. https://huggingface.co/api/models?filter=dataset:shibing624/medical&full=true - 18 models declaring this dataset in their tags, with their downloads/likes. Fetched 2026-08-11.

[8] The corpus screening row for `shibing624/medical`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[9] datasets-server size endpoint for the upstream corpus. https://datasets-server.huggingface.co/size?dataset=FreedomIntelligence%2Fhuatuo_encyclopedia_qa - live row counts by split, for the Neighbors and Quality comparisons. This endpoint takes no revision parameter, so this count is live, not pinned. Fetched 2026-08-11.

[10] shibing624/huatuo_medical_qa_sharegpt dataset card and Hub API record. https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt/raw/main/README.md and https://huggingface.co/api/datasets/shibing624/huatuo_medical_qa_sharegpt?full=true - upstream sources, `wc -l` counts, column schema (`conversations`), siblings list. Fetched 2026-08-11.

[11] Hugging Face Hub datasets search API. https://huggingface.co/api/datasets?search=medical_book_zh&full=true and https://huggingface.co/api/datasets?search=huatuo_encyclopedia_qa&full=true - searched for a mirror or re-release of this exact repository; none found beyond the upstream huatuo_encyclopedia_qa itself and its own re-hosts. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, with the config split made explicit: pick `finetune` for SFT pairs and `reward` for preference triples, hold out each config's own `validation`/`test` split, and treat the `pretrain` config with the caution this card sets out - both its unserved `medical_book_zh.json` file and its unconfirmed overlap with part of `finetune`'s Chinese train data (see Hold out and Quality) [1][2][9]. This matches the screening row's own guidance to use the finetune and reward parts and leave the pretrain part alone [8].

### The screening row

The row's own note [8]: "A Chinese medical collection in three separate parts: a finetune/ SFT set with train, valid and test json for Chinese and English, a pretrain/ encyclopedia corpus of about 360k entries plus 8,475 medical-textbook passages, and a reward/ preference set with its own train, valid and test; apache-2.0, and the viewer returns 501 because the repo is a loading script, so a run should take the finetune and reward parts and leave the pretrain part alone." The row carries no flag.
