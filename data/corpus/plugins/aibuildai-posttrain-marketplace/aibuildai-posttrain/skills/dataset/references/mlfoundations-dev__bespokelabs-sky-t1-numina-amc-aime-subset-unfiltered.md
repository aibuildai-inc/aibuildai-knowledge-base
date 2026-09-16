# mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered

4,069 math-competition problems, each carrying a DeepSeek-R1 reasoning trace and answer next to NuminaMath's original human ground-truth solution and answer, with no correctness filtering applied.

**mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered** is a pre-filter intermediate artifact from a reproduction of the Sky-T1 reasoning-distillation data pipeline: it takes the `amc_aime`-source problems from AI-MO's NuminaMath-CoT corpus [1] and generates a DeepSeek-R1 reasoning trace and final answer for each one, following the pipeline Bespoke Labs describes for building Bespoke-Stratos-17k, itself a port of Berkeley Sky-T1's data-curation recipe into the Bespoke Curator tool [2][3]. That recipe's next step, rejection sampling, discards traces whose extracted answer disagrees with the ground truth [4]; this repository's name says that step has **not** been run - both correct and incorrect DeepSeek-R1 traces remain, so a reader must filter on `deepseek_final_answer` vs. `ground_truth_final_answer` (or use the already-judged sibling below) before treating any row as a correct-reasoning SFT example. It lives at https://huggingface.co/datasets/mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered . **Because the source pool is NuminaMath's `amc_aime` subset - AMC and AIME competition problems - training on this data risks overlap with AIME-derived evaluation sets; the corpus screening flag names this specifically as an AIME 2025 lineage risk, the same one-hop case noted for OpenR1's calibration data [9].**

**Use it for**: reasoning-trace SFT data, but only after filtering for correctness yourself (compare `deepseek_final_answer` to `ground_truth_final_answer`, or fetch the already-filtered `correct` column from the sibling described under Neighbors) - never train on it unfiltered. Restriction: **do not** use for training a model you plan to score on AIME 2025 or other AMC/AIME-derived benchmarks, since the source pool shares that competition lineage [9]. Rows map to the reasoning-trace SFT format used by the reasoning-SFT method card (problem/instruction plus a reasoning-and-answer completion), not to any preference or chat-template format.

**Licence**: not stated - the repository's YAML front matter carries no `license` field and its tags list no `license:` entry [5][6]. Upstream NuminaMath, the problem source, is Apache-2.0 [1]; no source states a licence for the DeepSeek-R1-generated `reasoning`/`deepseek_solution`/`deepseek_final_answer` columns added here.

**Shape**: one config (`default`), one split (`train`), 4,069 rows, 6 string columns [6][7].

**Hold out**: no internal split to hold out (the repository has only `train`). The risk to manage is not a row range inside this file but the AMC/AIME-lineage overlap named above: hold out AIME-derived evaluation sets from any model trained on this pool, rather than holding out rows within it [9].

**Origin**: built by mlfoundations-dev; problems and ground-truth solutions come from NuminaMath's human-collected `amc_aime` subset, and the `reasoning`/`deepseek_solution`/`deepseek_final_answer` columns are DeepSeek-R1 generations [1][3]. Hub API at the check date: 587 downloads, 0 likes [5].

**Trained-on-by**: none found - no source states that any released model was trained on this specific unfiltered, per-subset intermediate. The pipeline it feeds is documented: Bespoke Labs used the merged, rejection-sampled version of this and two sibling unfiltered subsets (see Neighbors) to help build Bespoke-Stratos-17k, which trained Bespoke-Stratos-32B and Bespoke-Stratos-7B [2][3][8].

**Introduced by**: no paper - the dataset's Hub API record carries no `description` field, and its README is a bare YAML schema block with no descriptive prose below it [5]. The pipeline that produced it is described in the Bespoke Labs blog post [3] and the Sky-T1 blog post [4].

## Shape

Rows and columns, from the datasets-server `/size` and `/info` endpoints [6][7]:

| split | rows | columns |
| --- | --- | --- |
| `train` | 4,069 | 6 |

| column | dtype |
| --- | --- |
| `problem` | string |
| `reasoning` | string |
| `deepseek_solution` | string |
| `ground_truth_solution` | string |
| `deepseek_final_answer` | string |
| `ground_truth_final_answer` | string |

Original Parquet download size is 35,057,071 bytes; decoded in-memory size is 76,697,822 bytes per the live `/size` endpoint [6] (the repository's own YAML states a slightly higher decoded figure, 77,618,466 bytes [5]; the two differ by about 1.2% and no source explains the gap). No source states sequence-length or token statistics for this release.

## Quality

- The dataset is deliberately unfiltered: the pipeline it belongs to discards a DeepSeek-R1 (or, in the original Sky-T1 recipe, QwQ) trace whenever its extracted final answer disagrees with the problem's ground-truth answer, and this repository is the stage before that rejection-sampling step runs [3][4].
- Reading all 13 rows the datasets-server first-rows endpoint returns for `train` at offset 0, only 5 of 13 rows carry a short, clean value in both `deepseek_final_answer` and `ground_truth_final_answer` (e.g. `"2"`, `"6"`, `"0.5"`); the other 8 rows have the entire (whitespace-stripped) solution text dumped into one or both of those fields instead of a short answer, showing the automated answer-extraction that produced these two columns fails outright on a majority of this small sample [7]. Among the 5 rows with clean short answers, only 1 has `deepseek_final_answer` equal to `ground_truth_final_answer`; the other 4 disagree [7]. Both symptoms - extraction failures and genuine wrong answers - match what motivated Bespoke Labs to replace Sky-T1's regex/sympy answer parser with an LLM judge, which raised their retained-correct rate from 25% to 73% [3]. This 13-row read cannot be generalized to the full 4,069-row split; it establishes only that the two `_final_answer` columns are not reliable as-is for filtering without either re-parsing or an LLM judge.
- No source states a duplicate-row rate or an annotator-agreement figure for this repository.

## Load it

```python
import datasets

REV = "5171d850ba26e8b0a69e0b2258f9e22ad865b9c6"  # main at the check date
ds = datasets.load_dataset(
    "mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered",
    revision=REV, split="train",
)  # 4,069 rows, all unfiltered
```

**Trap**: nothing in `deepseek_final_answer` or `ground_truth_final_answer` marks a row correct or incorrect - training on the raw file as SFT data will teach the model some fraction of wrong-answer traces, and the two answer columns cannot always be string-compared directly to find them, since (per the Quality read above) they sometimes hold dumped solution text rather than a short answer. Either re-run answer extraction and matching yourself, or load the already-judged sibling below instead of comparing these two columns as they stand.

## Neighbors

- `mlfoundations-dev/bespokelabs-sky-t1-numina-rejection-sampled` - the merged, judged output this pool feeds: 39,045 rows, which is exactly the sum of this repository's 4,069 rows with two sibling unfiltered subsets' 14,976 and 20,000 rows (4,069 + 14,976 + 20,000 = 39,045) [10][11][12]. It adds `correct` (bool) and `judge_reasoning` (string) columns not present here, i.e. the LLM-judge correctness pass has already run [10]. Prefer this sibling over the raw unfiltered pool for any SFT use, since it already carries the correctness label this pool leaves you to compute.
- `mlfoundations-dev/bespokelabs-sky-t1-numina-math-subset-unfiltered` - the same unfiltered, un-judged shape, 14,976 rows, over NuminaMath's `math`-source problems instead of `amc_aime` [11].
- `mlfoundations-dev/bespokelabs-sky-t1-numina-olympiads-subset-unfiltered` - same shape, 20,000 rows, over NuminaMath's `olympiads`-source problems [12].
- `bespokelabs/Bespoke-Stratos-17k` - the actual training set Bespoke Labs released and used to fine-tune Bespoke-Stratos-32B and -7B: 17k examples spanning math (from these NuminaMath AIME/MATH/Olympiads subsets, after rejection sampling), coding (APPs/TACO), and science/puzzle (STILL-2) data; Apache-2.0 licensed [8]. This is the dataset to use for reproducing that training run, not the raw per-subset pool.
- `NovaSky-AI/Sky-T1_data_17k` - the original Sky-T1 recipe's released 17k-row training set, built with QwQ-32B-Preview as the reasoning model instead of DeepSeek-R1, over the same kind of NuminaMath AIME/MATH/Olympiads mix plus APPs/TACO and STILL-2; Apache-2.0 licensed [13][14]. Use this if you want the original recipe's teacher model rather than DeepSeek-R1.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], with `reasoning` truncated:

```json
{
  "problem": "The operation $\\otimes$ is defined for all nonzero numbers by $a\\otimes b =\\frac{a^{2}}{b}$. Determine $[(1\\otimes 2)\\otimes 3]-[1\\otimes (2\\otimes 3)]$.\n$\\text{(A)}\\ -\\frac{2}{3}\\qquad\\text{(B)}\\ -\\frac{1}{4}\\qquad\\text{(C)}\\ 0\\qquad\\text{(D)}\\ \\frac{1}{4}\\qquad\\text{(E)}\\ \\frac{2}{3}$",
  "reasoning": "Okay, let me try to figure out this problem. So, we have this operation defined as a⊗b = a²/b. And we need to compute [(1⊗2)⊗3] - [1⊗(2⊗3)]. [...] Therefore, I think my initial calculation was correct, and the answer is option A.\n\n**Final Answer**\n\\boxed{A}",
  "deepseek_solution": "To determine the value of \\([(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]\\) [...] Final",
  "ground_truth_solution": "1. **Apply the operation $\\otimes$ to the innermost parentheses first:** [...] \\left(\\frac{1}{12}\\right) - \\left(\\frac{3}{4}\\right) = \\frac{1}{12} - \\frac{9}{12} = -\\frac{8}{12} =",
  "deepseek_final_answer": "2",
  "ground_truth_final_answer": "2"
}
```

Note that row 0's `deepseek_final_answer`/`ground_truth_final_answer` are both the clean short value `"2"` (an answer-option index, not the letter `A` that appears inside `reasoning`); as the Quality section shows, most of the other 12 rows in this same first-rows read do not have such clean values in those two fields.

## Where it came from

Built by mlfoundations-dev. The `problem` and `ground_truth_solution`/`ground_truth_final_answer` columns come from the `amc_aime` source slice of AI-MO's NuminaMath-CoT dataset, a roughly 860k-problem corpus of OCR'd, translated, and reformatted competition and exam math problems whose `amc_aime` source alone holds about 4,072 samples [1]. The `reasoning`, `deepseek_solution`, and `deepseek_final_answer` columns are generated by DeepSeek-R1 as the teacher reasoning model, following a data-curation recipe Bespoke Labs ported from Berkeley Sky-T1 into their Bespoke Curator tool: Sky-T1 itself generated reasoning traces with QwQ-32B-Preview and discarded (rejection-sampled) any trace whose extracted math answer did not exact-match the dataset's ground truth [4]; Bespoke Labs reproduced this with DeepSeek-R1 as the teacher and, for the correctness check, an LLM judge (gpt-4o-mini) in place of Sky-T1's regex/sympy parser [3]. This repository represents that pipeline's per-subset output before the rejection-sampling/judging step runs; the judged, merged version across this and two sibling subsets is `mlfoundations-dev/bespokelabs-sky-t1-numina-rejection-sampled` [10].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hugging Face Hub repositories are mutable, which is why Load it pins the revision (commit `sha`) read at the check date.

[1] AI-MO/NuminaMath-CoT dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-CoT - Apache-2.0 licence, dataset summary, collection method (OCR, translation, realignment to CoT), and the `amc_aime` source-breakdown table (4,072 samples). Fetched 2026-08-11.

[2] bespokelabs/Bespoke-Stratos-17k dataset card (README). https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k - what the 17k dataset is, what it trained, and that it "replicated and improved" the Berkeley Sky-T1 pipeline using DeepSeek-R1. Fetched 2026-08-11.

[3] Bespoke Labs blog, "Bespoke-Stratos: The unreasonable effectiveness of reasoning distillation". https://bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation - porting the Sky-T1 pipeline into Bespoke Curator with DeepSeek-R1, using gpt-4o-mini in place of Sky-T1's parsing logic and raising the retained-correct rate from 25% to 73%. Fetched via its permanent redirect target 2026-08-11.

[4] NovaSky-AI (Berkeley Sky Computing Lab) blog, "Sky-T1: Fully open-source reasoning model with o1-preview performance". https://novasky-ai.github.io/posts/sky-t1/ - the data-curation recipe: QwQ-32B-Preview reasoning generation, rejection sampling by exact match to ground truth for math, and the final 5k-code/10k-math(AIME/MATH/Olympiads of NuminaMATH)/1k-science mixture. Fetched 2026-08-11.

[5] Hugging Face Hub API record for this dataset. https://huggingface.co/api/datasets/mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered?full=true - licence (absent), gate status, `sha`, `dataset_info` YAML, downloads, likes, last-modified date, and no `description` field, corroborated by the raw README (https://huggingface.co/datasets/mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered/raw/main/README.md), which is a 518-byte YAML front-matter block with no body prose below the closing `---`. Fetched 2026-08-11.

[6] datasets-server size and info endpoints for this dataset. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fbespokelabs-sky-t1-numina-amc-aime-subset-unfiltered and https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Fbespokelabs-sky-t1-numina-amc-aime-subset-unfiltered - row counts, byte sizes, column dtypes. Fetched 2026-08-11.

[7] datasets-server first-rows endpoint for this dataset. https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2Fbespokelabs-sky-t1-numina-amc-aime-subset-unfiltered&config=default&split=train - the 13 sampled rows used for the Quality read and the one row shown under A row. Fetched 2026-08-11.

[8] bespokelabs/Bespoke-Stratos-17k README, again: the 5k-code/10k-math/1k-science composition and the models it trained (Bespoke-Stratos-32B, Bespoke-Stratos-7B). Same source as [2]. Fetched 2026-08-11.

[9] The corpus screening row for `mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered`, supplied with this card's request - its `flag` ("rule-risk: aime2025 - Built from NuminaMath's `amc_aime` problem pool; same one-hop AIME lineage as the OpenR1 calibration case"), read back in the appendix. Checked 2026-08-11.

[10] datasets-server size and info endpoints for `mlfoundations-dev/bespokelabs-sky-t1-numina-rejection-sampled`. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fbespokelabs-sky-t1-numina-rejection-sampled and the matching `/info` endpoint - 39,045 rows, 8 columns including `correct` and `judge_reasoning`; live, not revision-pinned. Fetched 2026-08-11.

[11] datasets-server size endpoint for `mlfoundations-dev/bespokelabs-sky-t1-numina-math-subset-unfiltered`. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fbespokelabs-sky-t1-numina-math-subset-unfiltered - 14,976 rows; live, not revision-pinned. Fetched 2026-08-11.

[12] datasets-server size endpoint for `mlfoundations-dev/bespokelabs-sky-t1-numina-olympiads-subset-unfiltered`. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fbespokelabs-sky-t1-numina-olympiads-subset-unfiltered - 20,000 rows; live, not revision-pinned. Fetched 2026-08-11.

[13] NovaSky-AI/Sky-T1_data_17k dataset card (README) and Hub API record. https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k and https://huggingface.co/api/datasets/NovaSky-AI/Sky-T1_data_17k?full=true - dataset description (17k rows: 5k APPs/TACO code, 10k NuminaMATH-subset math, 1k STILL-2 science/puzzle) and Apache-2.0 licence. Fetched 2026-08-11.

[14] Same NovaSky-AI blog as [4], for the QwQ-32B-Preview teacher-model detail attributed to the original Sky-T1 recipe. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable only after filtering, and only outside any AIME-benchmark evaluation pipeline. The card's own read of the pipeline this repository belongs to shows rejection sampling has not yet run here, so wrong-answer traces remain, and the answer columns needed to filter them are themselves unreliable on part of the sample read above [3][4][7]; separately, the source pool's AMC/AIME competition lineage carries a contamination risk against AIME-derived evaluations, per the screening row's flag [9].

### The screening row

The row's own note [9]: "Sky-T1 build: Numina amc_aime problems with DeepSeek reasoning beside the human ground-truth solution, unfiltered so wrong answers remain." Its flag: "rule-risk: aime2025 - Built from NuminaMath's `amc_aime` problem pool; same one-hop AIME lineage as the OpenR1 calibration case."
