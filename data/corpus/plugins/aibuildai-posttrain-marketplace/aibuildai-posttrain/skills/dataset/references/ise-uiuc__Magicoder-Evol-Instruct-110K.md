# ise-uiuc/Magicoder-Evol-Instruct-110K

111,183 single-turn instruction/response pairs for code - a decontaminated republish of another team's GPT-4-generated Evol-Instruct-style coding data, released by the Magicoder authors alongside their own OSS-Instruct corpus.

**ise-uiuc/Magicoder-Evol-Instruct-110K** is a decontaminated copy of `theblackcat102/evol-codealpaca-v1` [1], itself an open-source reproduction of WizardCoder's Evol-Instruct method built by augmenting `HuggingFaceH4/CodeAlpaca_20K` with GPT-4 (`gpt-4-0314` and `gpt-4-0613`) [1]. The Magicoder paper "Magicoder: Empowering Code Generation with OSS-Instruct" [2] uses this file to continue-finetune its OSS-Instruct-trained models into the stronger MagicoderS series, and describes decontaminating evol-codealpaca-v1 by removing exact matches against HumanEval, MBPP, DS-1000, and GSM8K, filtering out 89 problems [2]. This repository's own card states its decontamination was done the same way as StarCoder's, via the bigcode decontamination process [3]. It lives at https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K . **The upstream `evol-codealpaca-v1` card warns that the same questions recur in `teknium/OpenHermes-2.5`, so deduplicate against that corpus before a scored run that includes it [1].**

**Use it for**: single-turn SFT on code instructions - each row is an `instruction`/`response` string pair, i.e. a prompt-completion pair with no chat template (`chat_dialect: none`), matching the SFT method card's plain prompt-completion input shape. No preference or reasoning-trace structure is present.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tags include `license:apache-2.0`) [4]. Ungated, public (`"gated": false`, `"private": false`) [4]. The one catch: the license covers this repository's own release, not the GPT-4 outputs' upstream terms-of-use status, which no source here addresses.

**Shape**: 111,183 rows, one split (`train`), one config (`default`), two string columns (`instruction`, `response`); 255,186,224 bytes as the original JSONL download, 136,512,547 bytes as Parquet [5][6].

**Hold out**: nothing named as an explicit split to hold out on this card; the contamination risk instead runs the other way - decontaminate against HumanEval, MBPP, DS-1000, GSM8K before a scored run using them, per the bigcode-style process the card names [3] and the paper's own decontamination step against the same four benchmarks [2]. Also deduplicate against `teknium/OpenHermes-2.5`, which the upstream card says shares the same questions [1].

**Origin**: released by the Magicoder authors (ise-uiuc) as a decontaminated redistribution of `theblackcat102/evol-codealpaca-v1` [1][3]; both `instruction` and `response` are GPT-4 generations from that upstream project, not human-written or human-labeled [1]. Hub API at the check date: `downloads` 14,397, `downloadsAllTime` 243,952, `likes` 187 [4].

**Trained-on-by**: the paper's own MagicoderS-CL-7B and MagicoderS-DS-6.7B, obtained by continuing to finetune the OSS-Instruct-trained Magicoder-CL-7B and Magicoder-DS-6.7B models on this file [2]; those four models (`ise-uiuc/Magicoder-CL-7B`, `ise-uiuc/Magicoder-S-CL-7B`, `ise-uiuc/Magicoder-DS-6.7B`, `ise-uiuc/Magicoder-S-DS-6.7B`) are the only ones the author's own Hub listing shows [7]. No adoption outside the Magicoder authors' own models is found.

**Introduced by**: [2] (Wei et al.), which names and decontaminates this file as `evol-codealpaca-v1`; the redistribution itself carries no separate paper, only this repository's dataset card [3].

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 111,183 |

One config, `default`, with two columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `response` | string |

Sizes (datasets-server `/size`) [5]: 255,186,224 bytes of original JSONL download, 136,512,547 bytes as Parquet, 246,198,605 bytes decoded in memory. No source states sequence-length or token statistics for this 111,183-row file. The upstream `evol-codealpaca-v1` card does state one figure for its own (111,272-row, pre-decontamination) release: a median sequence length of 471 [1]; no source restates this figure for the 89-row-smaller decontaminated copy here.

## Quality

- Both `instruction` and `response` are GPT-4 generations (`gpt-4-0314` and `gpt-4-0613`) produced by the upstream Evol-Instruct reproduction, not human-authored or human-labeled [1].
- The paper states this repository's row count is the upstream's 111,272 rows minus 89 problems removed as exact matches against HumanEval, MBPP, DS-1000, and GSM8K [2]; 111,272 − 89 = 111,183, matching this repository's row count exactly [2][5].
- This repository's card additionally names the bigcode/StarCoder decontamination process as the method used, without stating a separate removed-row count for that process [3].
- No source states a duplicate-rate or annotator-agreement figure; none is invented here. The upstream card's own quality note is that a 2023-08-26 update filtered results to pure-English instructions and removed mentions of being trained by OpenAI [1].

## Load it

One split, no config argument needed; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-12-28) [4]. The `load_dataset` call below is pinned to that revision, but the row count, column info, and sampled row elsewhere on this card were read from the datasets-server `/size`, `/info`, and `/first-rows` endpoints, none of which accept a revision parameter [5][6][11]; those numbers are live reads of `main` as of the check date, not covered by the pin below.

```python
import datasets

REV = "b0079beaa0361d82412520b873715bee59cc7dd4"  # main at the check date
train = datasets.load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", revision=REV, split="train")  # 111,183 rows
```

**Trap**: the repository's only data file is `data-evol_instruct-decontaminated.jsonl`, listed in the repo tree but not converted to the standard Parquet layout most other Hub datasets use [8]; `load_dataset` still resolves it correctly, but tools that expect a `data/` Parquet directory (rather than a root-level `.jsonl`) may not find it without this exact loader call.

## Neighbors

- `theblackcat102/evol-codealpaca-v1` - the pre-decontamination upstream this file was built from, 111,272 rows against this repository's 111,183 [1][9]. Prefer this decontaminated copy over the raw one for any run scored against HumanEval, MBPP, DS-1000, or GSM8K.
- `ise-uiuc/Magicoder-OSS-Instruct-75K` - the Magicoder authors' other, distinct release: 75,197 rows generated by `gpt-3.5-turbo-1106` under the orthogonal OSS-Instruct method rather than Evol-Instruct, with a different, seven-column schema and an MIT license rather than Apache-2.0 [10]. The paper trains base Magicoder models on this file first, then continues finetuning on the Evol-Instruct file here to reach MagicoderS [2] - the two are meant to be used together in that order, not as alternatives.
- `theblackcat102/evol-code-zh` - a Chinese-language Evol-Instruct variant from the same upstream project, per that project's own card [1]; reach for it only if a Chinese-language corpus is needed.
- The upstream card also flags overlapping questions in `teknium/OpenHermes-2.5` [1]; that is a deduplication concern, not a preferred neighbor.

## A row

One config and one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [11]:

```json
{
  "instruction": "Please amend the subsequent Python script so that it includes a 'while' loop rather than the existing 'for' loop, which iterates through the items of an integer list.\n\nThe script currently has a bug where it attempts to print an object that is outside the bounds of the list. Fix this error and modify the script to use 'while' instead of 'for' loop. Ensure your script correctly handles empty lists. \n\n```python\n  # Establish an integer list\n  arr = [1, 2, 3, 4]\n\n  # Determine the length of the list\n  n = len(arr)\n\n  # Traverse the list and output each individual element\n  for i in range(n+1):\n      print(arr[i])\n```",
  "response": "```python\n# Establish an integer list\narr = [1, 2, 3, 4]\n\n# Determine the length of the list\nn = len(arr)\n\n# Initialize index at 0\ni = 0\n\n# Traverse the list and output each individual element\nwhile i < n:\n    print(arr[i])\n    i += 1\n```\nIn the given code, it tries to access `arr[n]` which is out of bounds as python list indexing starts from 0 and ends at n-1 for a list of length n. [...]"
}
```

`instruction` states a coding task, often embedding a code block to fix or extend; `response` is prose explanation plus one or more fenced code blocks. Of the first three served rows, instructions range from a short debugging request to a garbled OCR-mangled code dump the model is asked to fix, so instruction length and shape vary considerably even within this single schema [11].

## Where it came from

The upstream `theblackcat102/evol-codealpaca-v1` project applied ten augmentation strategies (following a WizardCoder-like Evol-Instruct method, done as an open-source reproduction) to `HuggingFaceH4/CodeAlpaca_20K`, then used `gpt-4-0314` and `gpt-4-0613` to answer each evolved instruction, with most generation done by `gpt-4-0314` [1]. This repository takes that 111,272-row upstream file and decontaminates it "in the same way as StarCoder", pointing to the bigcode-project decontamination process [3]; the Magicoder paper independently reports removing 89 exact matches against HumanEval, MBPP, DS-1000, and GSM8K from evol-codealpaca-v1, which is the exact gap between the upstream's 111,272 rows and this repository's 111,183 [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] theblackcat102/evol-codealpaca-v1 dataset card (README). https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1/raw/main/README.md - build method, GPT-4 model versions, median sequence length, OpenHermes-2.5 overlap warning, evol-code-zh pointer. Fetched 2026-08-11.

[2] Wei, Wang, Liu, Ding, Zhang, "Magicoder: Empowering Code Generation with OSS-Instruct", 2023. https://arxiv.org/abs/2312.02120 - names evol-codealpaca-v1, its decontamination against HumanEval/MBPP/DS-1000/GSM8K removing 89 rows, and MagicoderS training on it after OSS-Instruct. Read via ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2312.02120). Current title read from the live abs page. Fetched 2026-08-11.

[3] ise-uiuc/Magicoder-Evol-Instruct-110K dataset card (README). https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/raw/main/README.md - states this is a decontaminated version of evol-codealpaca-v1, decontaminated the same way as StarCoder via the bigcode decontamination process. Fetched 2026-08-11.

[4] Hugging Face Hub API record for ise-uiuc/Magicoder-Evol-Instruct-110K. https://huggingface.co/api/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K?full=true - license, gate, private status, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ise-uiuc%2FMagicoder-Evol-Instruct-110K - this endpoint takes no revision parameter, so the row and byte counts read from it are live reads of `main`, not covered by the Load it revision pin. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=ise-uiuc%2FMagicoder-Evol-Instruct-110K - this endpoint takes no revision parameter, so the column info read from it is a live read of `main`, not covered by the Load it revision pin. Fetched 2026-08-11.

[7] Hugging Face Hub API model listing for the ise-uiuc organization. https://huggingface.co/api/models?author=ise-uiuc&limit=50 - lists the four Magicoder/MagicoderS models as the author's only published models. Fetched 2026-08-11.

[8] Hugging Face Hub API repo tree for ise-uiuc/Magicoder-Evol-Instruct-110K. https://huggingface.co/api/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/tree/main - lists `data-evol_instruct-decontaminated.jsonl` as the sole data file. Fetched 2026-08-11.

[9] datasets-server size endpoint for theblackcat102/evol-codealpaca-v1. https://datasets-server.huggingface.co/size?dataset=theblackcat102%2Fevol-codealpaca-v1 - this endpoint takes no revision parameter, so this row count is live, not pinned. Fetched 2026-08-11.

[10] ise-uiuc/Magicoder-OSS-Instruct-75K dataset card and size endpoint. https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=ise-uiuc%2FMagicoder-OSS-Instruct-75K - generating model, license, row and column counts. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=ise-uiuc%2FMagicoder-Evol-Instruct-110K&config=default&split=train - this endpoint takes no revision parameter, so the sampled rows in "A row" are a live read of `main`, not covered by the Load it revision pin. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for single-turn code SFT: the dataset is a decontaminated republish of a GPT-4-generated Evol-Instruct-style corpus, and the screening row's own note prefers this copy over the raw upstream file precisely because of that decontamination [2][3]. No split needs holding out; the risk to manage instead is decontaminating against the benchmarks named above before a scored run, and deduplicating against `teknium/OpenHermes-2.5` per the upstream card's own warning [1].

### The screening row

The row's own note: "the same GPT-4 evol-codealpaca data after StarCoder-style decontamination; prefer this copy over the raw one (see FLAG)." The row carries no `flag` field.
