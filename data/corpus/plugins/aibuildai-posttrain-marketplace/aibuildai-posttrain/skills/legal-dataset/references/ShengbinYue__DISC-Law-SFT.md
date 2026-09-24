# ShengbinYue/DISC-Law-SFT

About 286,000 Chinese legal SFT rows in four JSONL files - consultation Q&A, judgment prediction, reading comprehension, summarisation and judicial-exam questions - from Fudan's DISC-LawLLM; its exam rows contain LawBench and AGIEval JEC-QA test items.

**ShengbinYue/DISC-Law-SFT** is the supervised fine-tuning set behind DISC-LawLLM, introduced in "DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services" by Yue et al. at Fudan University [1]. Its card describes two subsets: DISC-Law-SFT-Pair, which "aims to introduce legal reasoning abilities", and DISC-Law-SFT-Triplet, which "helps enhance the model's capability to utilize external legal knowledge" by pairing each answer with the statutes it relies on [2]. The card's table totals 403K rows including 108K of general data (Alpaca-GPT4 and Firefly), and says "We currently open-source most of the DISC-Law-SFT Dataset" [2]. It lives at https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT .

**The Pair file contains every item of LawBench tasks 1-2 and 3-6 and 902 of the 1,000 AGIEval JEC-QA-KD items at 50% or more n-gram coverage. Do not report those benchmarks after training on it without decontamination.**

**Use it for**: Chinese legal SFT across many task shapes. The Triplet files are the retrieval-grounded part: each row's `reference` lists the statute articles the answer cites [3], which suits training a model to answer from supplied law.

**Licence**: Apache 2.0 in the card metadata [4]. The card names Alpaca-GPT4 among the general data in the full mix [2], but the released files contain only legal rows (by `id` prefix) [3]. The one catch: much of the Pair data is generated from court documents and exam questions whose own terms the card does not address.

**Shape**: four JSONL files, 285,781 rows in total: `DISC-Law-SFT-Pair.jsonl` 166,758; `DISC-Law-SFT-Pair-QA-released.jsonl` 79,692; `DISC-Law-SFT-Triplet-QA-released.jsonl` 23,331; `DISC-Law-SFT-Triplet-released.jsonl` 16,000 [3]. Pair files have `id`, `input`, `output`; Triplet files add `reference`, a list of statute strings [3]. The viewer reports 0 rows from `/size` [5], though `/first-rows` serves the first file.

**Hold out**: no split is set aside. Measured against LawBench (5 tasks) and AGIEval JEC-QA-KD: `DISC-Law-SFT-Pair.jsonl` covers LawBench 1-2 500/500, LawBench 3-6 500/500, and JEC-QA-KD 902/1,000 items at 50% or more of their 15-character n-grams; the Triplet judgment file covers 20-21 items in each of LawBench 3-1, 3-3 and 3-4 [6]. Remove the `exam-*` rows before scoring on either benchmark.

**Origin**: mixed: court documents, exam questions and statutes, with answers partly written or rewritten by models, per the paper [1]. Hub API at the check date: `downloads` 784, `downloadsAllTime` 17,669, `likes` 180 [4].

**Trained-on-by**: the authors' `ShengbinYue/DISC-LawLLM` (34 downloads, 43 likes) and `ShengbinYue/LawLLM-7B` (63), plus community fine-tunes such as `zzy12222/Qwen2.5-7B-Instruct-law` and `bryandts/Qwen2.5-0.5B-Instruct-Disc-Law-SFT` [7].

**Introduced by**: [1] (Yue et al.).

## Shape

Rows per file, counted by downloading each file [3]:

| file | rows | columns | `id` prefixes |
| --- | --- | --- | --- |
| `DISC-Law-SFT-Pair.jsonl` | 166,758 | `id`, `input`, `output` | `jud_read` 38,530; `leg_ele` 32,042; `leg_eve` 21,289; `leg_case` 20,563; `sent` 11,657; `jud_doc` 8,234; `sim_case` 8,138; `op` 5,251; `exam-N` the rest |
| `DISC-Law-SFT-Pair-QA-released.jsonl` | 79,692 | `id`, `input`, `output` | `legal_question_answering` |
| `DISC-Law-SFT-Triplet-QA-released.jsonl` | 23,331 | `id`, `reference`, `input`, `output` | `legal_question_answering` |
| `DISC-Law-SFT-Triplet-released.jsonl` | 16,000 | `id`, `reference`, `input`, `output` | `judgement` |
| total | 285,781 | | |

## Quality

- Measured exact duplication after removing whitespace and punctuation: Pair 5,054 repeats (3.03%, largest group 55); Pair-QA 3,461 (4.34%); both Triplet files 0 [8].
- The released rows fall short of the card's 295K legal rows (403K less 108K general); the card itself says "most" of the data is released [2].
- The Pair file's `exam-N` rows are judicial-examination questions - the source of the JEC-QA and LawBench overlap measured above [6].

## Load it

The files have two schemas, so load them separately with `data_files`, pinned:

```python
import datasets

REV = "fb12cf02809a85724f7e36977529b1d4b5f9f920"  # main at the check date
pair = datasets.load_dataset("ShengbinYue/DISC-Law-SFT", data_files="DISC-Law-SFT-Pair.jsonl", revision=REV, split="train")          # 166,758 rows
triplet = datasets.load_dataset("ShengbinYue/DISC-Law-SFT", data_files="DISC-Law-SFT-Triplet-released.jsonl", revision=REV, split="train")  # 16,000 rows
pair = pair.filter(lambda r: not r["id"].startswith("exam-"))  # before scoring LawBench or JEC-QA
```

**Trap**: a plain `load_dataset("ShengbinYue/DISC-Law-SFT")` sees all four files as one `train` split with two different schemas (`reference` exists in only two of them) [3]; the dataset viewer's `/size` answers 0 rows for exactly this repository [5]. Load each file with `data_files`.

## Neighbors

- `doolayer/LawBench` and `hails/agieval-jec-qa-kd` - evaluation sets whose items this dataset's exam rows contain; see their cards.
- `china-ai-law-challenge/cail2018` - the judgment-prediction source behind LawBench 3-1, 3-3 and 3-4.

## A row

From `DISC-Law-SFT-Triplet-released.jsonl`, line 1, downloaded [3], truncated:

```json
{
  "id": "judgement_predit-1",
  "reference": ["《刑法》第一百一十四条：【放火罪】【决水罪】【爆炸罪】【投放危险物质罪】【以危险方法危害公共安全罪】放火、决水、爆炸以及投放毒害性、放射性、传染病病原体等物质或者以其他危险方法危害公共安全，尚未造成严重后果的，处三年以上十年以下有期徒刑。"],
  "input": "基于下列案件，推测可能的判决结果。\n经审理查明，2015年6月21日15时许，被告人白某某在大东区小河沿公交车站乘坐被害人张某某驾驶的133路公交车 [...]",
  "output": "根据《刑法》第一百一十四条的规定，被告人白某某以危险方法危害公共安全，尚未造成严重后果。 [...]"
}
```

## Where it came from

Built by the Fudan DISC lab for DISC-LawLLM from Chinese legal NLP datasets, judgments, exam questions and statutes, with model-assisted rewriting described in the paper [1]. The project homepage is https://github.com/FudanDISC/DISC-LawLLM [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Yue et al., "DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services", arXiv:2309.11325, 2023. https://arxiv.org/abs/2309.11325 - current title read from the live abs page. Fetched 2026-09-23.

[2] ShengbinYue/DISC-Law-SFT dataset card (README). https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT/raw/main/README.md. Fetched 2026-09-23.

[3] ShengbinYue/DISC-Law-SFT data files, downloaded from https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT/resolve/main/<file> and read line by line - row counts, keys, `id` prefixes and first rows. Fetched 2026-09-23.

[4] Hugging Face Hub API record for ShengbinYue/DISC-Law-SFT. https://huggingface.co/api/datasets/ShengbinYue/DISC-Law-SFT?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ShengbinYue%2FDISC-Law-SFT - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] This skill's own measurement, `references/contamination.md`, row "DISC" - n-gram containment with its three controls; method and script in that file - 15-character n-grams, eval items sampled to 200 n-grams each. Run 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:ShengbinYue/DISC-Law-SFT&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] This skill's duplication measurement on each full file, `references/contamination.md` (duplication table) - `input`+`output`(+`reference`) with whitespace and punctuation removed. Run 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as Chinese legal SFT data, especially the statute-grounded Triplet files. Its `exam-*` rows contain LawBench and AGIEval JEC-QA test items; strip them before reporting either benchmark. Load files separately because the schemas differ.

### The screening row

The row's own note: "Chinese legal SFT for DISC-LawLLM; exam rows overlap LawBench/JEC-QA." The row carries the flag `contains-benchmark-items`.
