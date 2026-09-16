# lavita/AlpaCare-MedInstruct-52k

52,002 medical instruction/input/output triples, machine-generated to fine-tune LLaMA-series models for medical instruction-following, in a single `train` split.

**lavita/AlpaCare-MedInstruct-52k** is a Parquet mirror of MedInstruct-52k, introduced in "AlpaCare:Instruction-tuned Large Language Models for Medical Application" [1] to instruction-tune the paper's AlpaCare models. Each row is an Alpaca-style triple - `instruction`, `input`, `output` - built by prompting GPT-4 with a 167-item clinician-crafted seed set to generate new medical tasks, removing duplicate generated tasks, and then having ChatGPT (GPT-3.5-turbo) write the response to each surviving task [1]. Comparing the first two rows served by this repository against the origin authors' own `data/MedInstruct-52k.json` file shows byte-identical `instruction`/`input`/`output` text, so this is the same 52k-row data reformatted to Parquet [2][3]. It lives at https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k . **This repository carries no license field on the Hub, but the origin authors' own release of the same data, and their own AlpaCare model weights trained on it, are both licensed CC BY-NC 4.0 (non-commercial) [2][4]; treat this mirror as inheriting that restriction.**

**Use it for**: SFT on instruction-following triples (Alpaca-style prompt/response format) - not preference pairs. Build the training prompt from `instruction` and `input` (roughly 35% of the first 100 served rows carry the literal placeholder string `<noinput>` rather than an empty string, so a template needs to special-case it) and supervise on `output`. Treat as **non-commercial use only** per the origin license (see above). Maps to the SFT method card.

**Licence**: not stated on this repository (`license` field null, no `license:` tag, no license text in the card body); the same content's origin repository states CC BY-NC 4.0, so treat this mirror the same way [2].

**Shape**: 52,002 rows, one config (`default`), one split (`train`), three string columns (`instruction`, `input`, `output`) [5][6].

**Hold out**: nothing found. No source states that this 52k-row set overlaps an evaluation set; the paper's own medical evaluation set, MedInstruct-test, is a separate 217-row file hosted in the origin repository, not part of this release [1][2].

**Origin**: builder is the Hub account "lavita", re-hosting the AlpaCare authors' data as Parquet; instructions are GPT-4 generations and responses are ChatGPT (GPT-3.5-turbo) generations, per the origin authors' own dataset card [1][2]. Hub API as of the check date: 1,679 downloads, 9,748 all-time downloads, 23 likes [7].

**Trained-on-by**: the paper's own AlpaCare LLaMA-1-7B, LLaMA-2-7B, LLaMA-13B and LLaMA-2-13B models, hosted at `xz97/AlpaCare-llama1-7b`, `xz97/AlpaCare-llama2-7b`, `xz97/AlpaCare-llama-13b` and `xz97/AlpaCare-llama2-13b`, whose model cards describe them as LLMs tuned on medical instructions [4]. No third-party adoption beyond the AlpaCare authors' own models is confirmed here.

**Introduced by**: [1] (Zhang et al.).

## Shape

Rows and splits, from the live datasets-server (no revision parameter; matches the pinned `cardData` counts) [5]:

| split | rows |
| --- | --- |
| `train` | 52,002 |

One config, `default`, three columns [6]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

The repository's own `cardData` (read at commit `fad8593f748de481046e54f39400d5282f695280`, the same as the live `main` `sha` at the check date) states a download size of 36,697,625 bytes and an in-memory/decoded dataset size of 64,721,846 bytes [8]. The live datasets-server `/size` and `/info` endpoints agree on the 36,697,625-byte download figure but report a different in-memory size, 69,800,048 bytes [5][6]; no source explains the 5,078,202-byte gap between the two in-memory figures. No source states sequence-length or token statistics for this release.

## Quality

- The generation process removes duplicate GPT-4-generated tasks before the response-generation step, per the origin card, but no source states a measured duplicate rate for the resulting 52k rows, nor any measured contamination or annotator-agreement figure [1][2].
- Reading the first 100 served rows at offset 0 (`config="default"`, `split="train"`): 35 of the 100 `input` fields hold the literal placeholder string `<noinput>` rather than being empty or absent; the rest hold task-specific context text (a patient note, a lab result, a passage to rewrite, and similar) [9].
- The response side is written entirely by ChatGPT (GPT-3.5-turbo), not by a clinician or a stronger model, per the origin card [2]; no source states a measured factual-accuracy rate for these machine-written responses.
- The paper's abstract reports the trained-on-this-data AlpaCare models beat the strongest baseline by up to 38.1 absolute points on its medical free-form instruction evaluations and by 6.7 absolute points averaged over its general-domain benchmarks, and that human evaluators rated AlpaCare ahead of the baselines on both correctness and helpfulness [1]. The paper's own Results section (4.1/4.2) presents these same comparisons only as figures (Figure 2(b), Figure 3, Figure 4), stating the outcome qualitatively in text ("AlpaCare consistently and significantly surpasses..."); no numeric value for these comparisons appears in that section's extracted text, so the 38.1/6.7 figures above are the abstract's own numbers, not a value read off the body's figures [1]. These are results of training on this dataset, not a property of the rows themselves, so they belong here as the "does it work" signal rather than in Shape.

## Load it

```python
import datasets

REV = "fad8593f748de481046e54f39400d5282f695280"  # main at the check date
ds = datasets.load_dataset("lavita/AlpaCare-MedInstruct-52k", revision=REV, split="train")  # 52,002 rows
```

**Trap**: there is only one split, `train` - there is no held-out validation or test split shipped with this repository, so any evaluation split must come from elsewhere (e.g. the origin authors' separate MedInstruct-test file) [2]. Also, `input` is a placeholder-bearing string, not an optional/nullable field - code that treats an empty `input` as "no input" will silently miss the ~35% of rows using the literal `<noinput>` string instead [9].

## Neighbors

All row counts below were read live at the check date [10].

- `xz97/MedInstruct` - the origin authors' own repository (point of contact: Xinlu Zhang, the paper's first author), bundling the same `data/MedInstruct-52k.json` this card mirrors, plus the 217-row `MedInstruct-test` evaluation file and the 167-item `med_seed.json` clinician seed set; its card states the license as CC BY-NC 4.0 and states the instructions/responses were written by GPT-4 and GPT-3.5-turbo respectively [2][10]. Its datasets-server auto-conversion only surfaces the 216-row `MedInstruct-test.jsonl` as a served config (the 52k file is not auto-converted to Parquet there), so use this `lavita` mirror, not `xz97/MedInstruct`, to load the 52k training rows programmatically [10].
- `li-lab/JP-AlpaCare-MedInstruct-52k` - 51,992 rows, a Japanese machine translation of this release (via `gpt-4o-2024-05-13`) that keeps both languages aligned in `instruction_en`/`instruction_ja` and similar column pairs. Its card is internally inconsistent on licensing: the YAML front matter tags it MIT, while the prose body says the license follows the original dataset (i.e. this release, whose own license this card treats as inherited CC BY-NC 4.0 - see above); no source resolves which of the two the repository actually intends [10][11].
- `adlbh/medinstruct-52k-arabic` - 52,002 rows, an Arabic-language variant with four columns [10].
- `ZhexiLu/AlpaCare-MedInstruct-52k` - 52,002 rows re-split by a third party into `train` (41,601) / `validation` (5,200) / `test` (5,201) over the same three columns, with no stated license or attribution text in its card; useful only if a pre-made train/val/test split of this exact data is wanted [10][11].

This corpus prefers this `lavita` mirror over `xz97/MedInstruct` because it is the one that actually serves the full 52k rows as a loadable Parquet split.

## A row

One served shape (`config="default"`, `split="train"`). Row `row_idx=0` [9]:

```json
{
  "output": "A mass in the lung could cause shortness of breath due to several reasons. First, the mass can physically obstruct the air passages, causing difficulty in airflow and leading to breathing difficulties. Second, if the mass is cancerous or infected, it can cause inflammation and damage to lung tissue, reducing its functional capacity and compromising normal breathing. Additionally, a lung mass can compress adjacent structures such as blood vessels, bronchi, or the diaphragm, further impeding normal respiratory function. Overall, any interference with the normal flow of air in the lungs caused by a mass can result in inadequate oxygen exchange and subsequent shortness of breath.\n\nThe answer is: A mass in the lung can obstruct air passages, cause inflammation, damage lung tissue, and compress adjacent structures, leading to shortness of breath.",
  "input": "<noinput>",
  "instruction": "Explain why a mass in the lung could cause shortness of breath."
}
```

## Where it came from

Built by the AlpaCare paper's authors and released on the Hub under `xz97/MedInstruct`; this `lavita` repository reformats the same 52k rows to Parquet [1][2][3]. The construction is a semi-automated, self-instruct-style pipeline: starting from 167 clinician-crafted seed tasks spanning multiple medical topics, task types, and difficulty levels, the authors prompt GPT-4 to generate 12 new candidate instructions at a time (each prompt drawing 3 random seed instructions), remove duplicate generated tasks, and then send each surviving instruction to ChatGPT (GPT-3.5-turbo) individually to synthesize the `output` response [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hub repositories are mutable, which is why Load it pins the revision.

[1] Zhang et al., "AlpaCare:Instruction-tuned Large Language Models for Medical Application", 2023. https://arxiv.org/abs/2310.14558 - the origin paper. Current title, and the 38.1%/6.7%/correctness-and-helpfulness performance claims, read from the live abs page's own abstract text; dataset construction and seed-set details read from the arXiv HTML full text (https://ar5iv.labs.arxiv.org/html/2310.14558), whose Results section (4.1/4.2) reports the same comparisons only via figures, with no matching numeric values in its extracted text. Fetched 2026-08-11.

[2] `xz97/MedInstruct` dataset card (README). https://huggingface.co/datasets/xz97/MedInstruct/raw/main/README.md - origin authors' own repository; license, generator attribution (GPT-4 for instructions, GPT-3.5-turbo for responses), seed-set size, MedInstruct-test description. Fetched 2026-08-11.

[3] `xz97/MedInstruct` raw data file, first bytes. https://huggingface.co/datasets/xz97/MedInstruct/resolve/main/data/MedInstruct-52k.json - read to confirm row-for-row content match against this repository's first served rows. Fetched 2026-08-11.

[4] `xz97/AlpaCare-llama2-7b` model card (README), and the sibling model listings `xz97/AlpaCare-llama1-7b`, `xz97/AlpaCare-llama-13b`, `xz97/AlpaCare-llama2-13b` from the Hugging Face Hub models-search API. https://huggingface.co/xz97/AlpaCare-llama2-7b/raw/main/README.md and https://huggingface.co/api/models?search=AlpaCare - license, "tuned on medical instructions" description. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=lavita%2FAlpaCare-MedInstruct-52k Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=lavita%2FAlpaCare-MedInstruct-52k Fetched 2026-08-11.

[7] Hugging Face Hub API record for lavita/AlpaCare-MedInstruct-52k, including the `expand[]=downloadsAllTime` variant. https://huggingface.co/api/datasets/lavita/AlpaCare-MedInstruct-52k?full=true - downloads, likes, gated status, `sha`, license field. Fetched 2026-08-11.

[8] Repository README front matter (`cardData`). https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k/raw/main/README.md - `dataset_info` byte and row counts. Fetched 2026-08-11.

[9] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=lavita%2FAlpaCare-MedInstruct-52k&config=default&split=train - 100 rows read at offset 0. Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor: `xz97/MedInstruct`, `li-lab/JP-AlpaCare-MedInstruct-52k`, `adlbh/medinstruct-52k-arabic`, `ZhexiLu/AlpaCare-MedInstruct-52k`. https://datasets-server.huggingface.co/size?dataset=<id> - live, unpinned, no revision parameter. Fetched 2026-08-11.

[11] Neighbor dataset cards (READMEs): `li-lab/JP-AlpaCare-MedInstruct-52k` and `ZhexiLu/AlpaCare-MedInstruct-52k`. https://huggingface.co/datasets/<id>/raw/main/README.md Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT instruction-following data, non-commercial use only. The dataset's own generation process (established above) yields Alpaca-style instruction/input/output triples suitable for supervised fine-tuning, and the screening row's note flags the generator-attribution gap this card resolves by going to the origin authors' own card [2].

### The screening row

The row's own note: "52k medical instructions the AlpaCare paper (arXiv 2310.14558) calls 'machine-generated'; neither the card nor the abstract names the generator." The row carries no flag.

The origin authors' own repository card does name the generators (GPT-4 for instructions, GPT-3.5-turbo for responses) even though this mirror's own card and the paper's abstract do not spell this out beyond "GPT-4 and ChatGPT" [1][2] - see the opening paragraph above.
