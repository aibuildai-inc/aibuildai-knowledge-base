# Rock23210/AIME_Deepseek_Clean

492 math-competition-style word problems, each paired with a long DeepSeek-style reasoning trace that ends in a boxed final answer, laid out as instruction/input/output triples.

**Rock23210/AIME_Deepseek_Clean** is a single-file Hub upload by the user `Rock23210`, at https://huggingface.co/datasets/Rock23210/AIME_Deepseek_Clean ; the repository's only card content is an MIT license front-matter field, with no description, author statement, or citation [1]. Every row carries the constant instruction "Please solve the following math problem.", a competition-style problem statement in `input`, and a worked solution in `output` that closes with a `\boxed{}` answer, matching the style of DeepSeek's chain-of-thought reasoning outputs [2]. **The repository does not state where its problems come from, and none of the 492 rows are labelled by year or number as they would be if drawn directly from a known AIME archive; a corpus check against AIME 2023, 2024 and 2025 problem sets found no matching rows despite the "AIME" name, but that negative result does not establish the problems are original or that they cannot overlap some other year's AIME set, so decontaminate against AIME benchmark splits before any scored evaluation run [3].**

**Use it for**: reasoning-trace SFT - each row is a full instruction/input/output triple ending in a worked, boxed answer, the shape the reasoning-SFT method card expects; not a preference or chat-turn format. Restriction: unverified problem provenance, so hold out or decontaminate against AIME benchmark splits before scoring on AIME [3].

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated, not private [1]. The one catch: the MIT grant covers Rock23210's redistribution of the file, not the underlying competition problems - the card names no problem source, so the license says nothing about where "AIME" comes from [1].

**Shape**: 492 rows, one split (`train`), one config (`default`), three string columns (`instruction`, `input`, `output`) [4][5].

**Hold out**: no rows are confirmed to overlap a known AIME set - a probe against AIME 2023, 2024 and 2025 matched 0 of the 492 rows [3] - but decontaminate against AIME benchmark splits before a scored run anyway, since the repository does not state a problem source and a negative probe on three years does not clear every possible source [1][3].

**Origin**: uploaded by Hub user `Rock23210`; no source dataset is declared in the repository metadata, and no generating model is named in the card - the "DeepSeek-style" reasoning-trace format is this card's own reading of the sampled rows, not a stated fact [1][2]. Hub API at the check date: 136 `downloads`, 1,089 `downloadsAllTime`, 0 `likes` [1].

**Trained-on-by**: none found - the Hub's model-listing endpoint filtered on this dataset id returns no models [6].

**Introduced by**: no paper - the dataset card [1], which itself carries only a license field and no description.

## Shape

One split, one config (datasets-server `/size` and `/info`) [4][5]:

| split | rows |
| --- | --- |
| `train` | 492 |

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

Every one of the 25 rows served at offset 0 carries the identical constant string in `instruction` ("Please solve the following math problem."), a non-empty `input`, and a non-empty `output` [2]. Byte sizes (datasets-server `/size`) [4]: 3,978,047 bytes as the original uploaded JSON, 1,496,594 bytes as Parquet, 3,641,475 bytes decoded in memory. No source states token counts; from the 25 rows read at offset 0, `output` ranges from 2,789 to 19,102 characters (mean about 6,658), and `input` for the first row is 316 characters - a character count, not a token count, and only over those 25 rows [2].

## Quality

No source states a measured accuracy, duplicate rate, or contamination rate for this dataset; none is invented here. From the 25 rows read at offset 0 [2]: every row's `output` contains a `\boxed{...}` final answer, and 15 of the 25 also contain the literal phrase "Final Answer" before it - consistent with a single generation style across at least that sample, though not confirmed dataset-wide. The card states no annotation, filtering, or "cleaning" process despite the repository name, and no source describes how the 492 problems were selected or verified [1].

## Load it

```python
import datasets

REV = "305a92216162de2bec5bfb415efa02e9889864c6"  # main at the check date
ds = datasets.load_dataset("Rock23210/AIME_Deepseek_Clean", revision=REV, split="train")  # 492 rows
```

**Trap**: the repository ships a single JSON file named `AIME Deepseek Clean.json` (with spaces) rather than a `data/` Parquet layout; `load_dataset` resolves this transparently via the Hub's JSON loader, but there is only one split (`train`) - there is no held-out split to load separately, so any decontamination against AIME benchmarks must be done by the caller after loading [1][7].

## Neighbors

- `di-zhang-fdu/AIME_1983_2024` - a 933-row benchmark of real AIME problems from 1983 to 2024, sourced from the Art of Problem Solving wiki and citable by its own DOI; its card states explicitly "Do not using in training!" and gives `ID`, `Year`, `Problem Number`, `Question`, `Answer`, `Part` columns rather than instruction/input/output pairs [8]. This is the kind of dated, sourced AIME archive that Rock23210/AIME_Deepseek_Clean's problems would need to be checked against, since this card names no year or problem number for any of its 492 rows [1][8].
- No sibling, cleaned, or successor release from the same uploader was found; the repository holds a single file and no other datasets by `Rock23210` were located during this check.

## A row

One shape is served. From `config="default"`, `split="train"`, `row_idx=1` (datasets-server `/first-rows`) [2], with the derivation truncated:

```json
{
  "instruction": "Please solve the following math problem.",
  "input": "Let $P(x)=2x^{59}-x^{2}-x-6$ . If $Q(x)$ is a polynomial whose roots are the $59$ th powers of the roots of $P(x)$ , then find the sum of the roots of $Q(x)$ .",
  "output": "Alright, I have this problem in front of me:\n\n*Let \\( P(x) = 2x^{59} - x^2 - x - 6 \\). If \\( Q(x) \\) is a polynomial whose roots are the 59th powers of the roots of \\( P(x) \\), then find the sum of the roots of \\( Q(x) \\).*\n\nFirst, I need to understand what's being asked. [...] \n\n**Final Answer:**\n\n\\(\\boxed{...}\\)"
}
```

## Where it came from

The repository is a single-file upload by Hub user `Rock23210`, last modified 2025-01-05 [1]. No source dataset is declared in the repository's metadata (`source_datasets` is empty), no upstream problem archive is named, and no generating model is credited anywhere in the card [1]. The problems read as competition-style word problems and the solutions read as long, step-by-step derivations closing in a boxed final answer, in the pattern of DeepSeek-style reasoning-model output, but this is this card's own characterization of the sampled rows, not a claim made by the repository itself [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hugging Face Hub API record for Rock23210/AIME_Deepseek_Clean. https://huggingface.co/api/datasets/Rock23210/AIME_Deepseek_Clean?full=true - license, gate, sha, downloads, likes, last-modified date, siblings, source_datasets; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant; card body confirmed as license-only via the raw README at https://huggingface.co/datasets/Rock23210/AIME_Deepseek_Clean/raw/main/README.md. Fetched 2026-08-11.

[2] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Rock23210%2FAIME_Deepseek_Clean&config=default&split=train - 25 rows read at offset 0. Fetched 2026-08-11.

[3] The corpus screening row for `Rock23210/AIME_Deepseek_Clean`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Rock23210%2FAIME_Deepseek_Clean Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Rock23210%2FAIME_Deepseek_Clean Fetched 2026-08-11.

[6] Hugging Face Hub models-list endpoint filtered on this dataset. https://huggingface.co/api/models?filter=dataset:Rock23210/AIME_Deepseek_Clean - returns an empty list. This endpoint takes no revision parameter, so this is a live check, not a pinned one. Fetched 2026-08-11.

[7] Repository tree listing. https://huggingface.co/api/datasets/Rock23210/AIME_Deepseek_Clean/tree/main - confirms the single file `AIME Deepseek Clean.json` plus `.gitattributes` and `README.md`. Fetched 2026-08-11.

[8] di-zhang-fdu/AIME_1983_2024 dataset card and datasets-server endpoints. https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/raw/main/README.md ; https://datasets-server.huggingface.co/size?dataset=di-zhang-fdu%2FAIME_1983_2024 ; https://datasets-server.huggingface.co/info?dataset=di-zhang-fdu%2FAIME_1983_2024 - row count, columns, "Do not using in training!" statement, AoPS source. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Flagged as rule-risk, usable only with the restriction stated above: the rows are self-declared "AIME" problems with generated solutions, the card names no problem source, and a probe against AIME 2023, 2024 and 2025 found no matching rows despite the name - the flag records this as an unresolved provenance gap rather than a confirmed contamination [1][3].

### The screening row

The row's own note [3]: "492 competition problems with DeepSeek-style generated solutions in instruction/input/output form; AIME 2023, 2024 and 2025 probes found nothing despite the name." Its flag [3]: "rule-risk: aime2025 - Self-declared AIME problems with generated solutions; licence-only card names no source, AIME items not removed (2025 probe matched 0)."
