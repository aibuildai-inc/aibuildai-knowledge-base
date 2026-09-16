---
description: >-
  Data that post-trains a base language model, and how to judge whether one is
  fit to use. 215 dataset cards, each carrying the licence, the exact columns
  and splits, a revision-pinned load line, a real sample row, sourced adoption
  evidence, and its screening record as an appendix. The body of this skill
  holds a short recommended set; references/index.md holds every dataset
  grouped by kind and marks the ones that are benchmarks and must be held out;
  references/finding-more.md is a live search procedure for when the list does
  not hold what is wanted. Use it when choosing what to train on, and before
  trusting any dataset a plan already names.
---

# Post-training data

Post-training data is the text a base language model is trained on after pretraining, and it comes in shapes: chat dialogues, instruction-answer pairs, preference pairs, and plain text. One thing decides a choice more than any other: the data has to match the shape of answer the grader reads, not only the topic. A perfect math corpus written as multiple-choice letters will not teach a model to show working that a step-graded benchmark wants. The second thing is contamination: a dataset that carries the evaluation's own items is worth nothing however good it looks, and the run that trains on it scores well and means nothing.

## When to consult this skill

- The designer, when the plan names a task and needs the data that trains for it.
- The implementer, before writing the load line: the card carries the exact revision, config, and split names, so a wrong split is caught before the download.
- Anyone building a training mix, to check that no part of it is a benchmark: 34 of the 215 datasets here are evaluation sets, and they are listed so they can be recognised and kept out.

## Recommended set

A short table, not the full list: 13 datasets the cards themselves show as the strongest for each training shape. Every other dataset is in `references/index.md`.

| Dataset | Shape | Rows | Licence | The one restriction | File |
|---|---|---|---|---|---|
| `allenai/tulu-3-sft-mixture` | SFT chat | 939,343 | odc-by | named subsets carry their own licences, and the No Robots subset is non-commercial | `references/allenai__tulu-3-sft-mixture.md` |
| `HuggingFaceH4/ultrachat_200k` | SFT chat | 515,311 | mit | hold out `test_sft` and `test_gen`; the `_gen` splits end on a user turn and are not direct SFT data | `references/HuggingFaceH4__ultrachat_200k.md` |
| `teknium/OpenHermes-2.5` | SFT chat | 1,001,551 | none on the card | no licence is stated anywhere, so the terms of every component are unresolved | `references/teknium__OpenHermes-2.5.md` |
| `HuggingFaceH4/no_robots` | SFT chat, human-written | 10,000 | cc-by-nc-4.0 | non-commercial use only | `references/HuggingFaceH4__no_robots.md` |
| `nvidia/OpenMathInstruct-2` | SFT reasoning traces | 21,972,791 | cc-by-4.0 | hold out GSM8K, MATH, AMC 2023, and AIME 2024; the Omni-MATH test set was never checked against | `references/nvidia__OpenMathInstruct-2.md` |
| `open-r1/OpenR1-Math-220k` | SFT reasoning traces | 450,258 | apache-2.0 | decontaminate the AMC and AIME-derived rows before a scored run | `references/open-r1__OpenR1-Math-220k.md` |
| `AI-MO/NuminaMath-CoT` | SFT reasoning traces | 859,594 | apache-2.0 | hold out its own `test` split (100 rows), then screen against post-2024 competition benchmarks, because its problem pools carry them | `references/AI-MO__NuminaMath-CoT.md` |
| `nvidia/OpenCodeInstruct` | SFT code, execution-verified | 1,400,000 served of 5,000,000 declared | cc-by-4.0 | the Hub copy is partial: it serves 1.4M of the 5M rows the card declares, so no whole-split number can be measured | `references/nvidia__OpenCodeInstruct.md` |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | SFT code | 111,183 | apache-2.0 | decontaminate against HumanEval, MBPP, DS-1000, and GSM8K before a scored run | `references/ise-uiuc__Magicoder-Evol-Instruct-110K.md` |
| `HuggingFaceH4/ultrafeedback_binarized` | preference pairs | 187,405 | mit | hold out all three test splits, and read the card: `messages` means a different thing in each split | `references/HuggingFaceH4__ultrafeedback_binarized.md` |
| `argilla/ultrafeedback-binarized-preferences-cleaned` | preference pairs | 60,917 | mit | there is no test split at all; build your own before training | `references/argilla__ultrafeedback-binarized-preferences-cleaned.md` |
| `nvidia/HelpSteer2` | reward-model data | 21,362 | cc-by-4.0 | hold out `validation` | `references/nvidia__HelpSteer2.md` |
| `Anthropic/hh-rlhf` | preference pairs | 169,352 | mit | preference-only: never train SFT on the chosen side, and hold out `test` (8,552 rows) | `references/Anthropic__hh-rlhf.md` |

The restriction cell is one clause, taken from the card's own opening. Everything beyond that clause is on the card and nowhere else: there is no second index to open.

## The full list

`references/index.md` holds all 215 datasets, grouped by kind, with the rows, the licence field, and whether the dataset may be trained on at all. Read it when the short table above has nothing that fits. `references/index.json` holds the same list with the measured columns, splits, and licence fields, for matching by field rather than by eye.

## Finding one that is not listed

`references/finding-more.md` is a procedure, not a list: which endpoint to call, in what order, and the traps that return a wrong answer with an HTTP 200. Use it when the index has no fit. The search really is needed, and not as a last resort: datasets appear, change, and disappear, and this list is one snapshot taken on 2026-08-11.

## How to read a card

A card exists only because the dataset passed screening, so it leads with what the dataset is and how to use it, and the screening record sits in an appendix at the end. Read the opening paragraph, the bolded `Use it for`, `Licence`, `Hold out`, and `Trap` lines, and stop there for most decisions. The load line under `Load it` is exact and pinned to the revision every number on the card was read at, so it can be copied straight into a script - there is no second file to open before acting.

## Scope

- The Hugging Face Hub, GitHub, and the DOI repositories (Zenodo, Harvard Dataverse, figshare, OSF, Dryad, Mendeley).
- The line is the platform, not the language of the data. A many-language corpus on a listed platform is in scope.
- A dataset whose Hub copy is marked `partial` is reported as not auditable, with its true row count taken from its card - never as a small dataset.
- Mendeley is unreached, not empty: every endpoint answered 403 behind a bot check, so its zero contributions mean nothing.
- **No signal here says whether an answer is correct.** That is the largest gap in this skill, and no first-hand reading closes it. A card can tell you who wrote the text and what shape it has; it cannot tell you the answers are right.
- Every claim on a card carries a URL and a commit or a fetch date. Nothing comes from our own runs.
