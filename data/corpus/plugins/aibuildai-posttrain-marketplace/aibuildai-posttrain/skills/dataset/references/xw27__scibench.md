# xw27/scibench

692 open-ended, free-response college-science problems - LaTeX problem text, LaTeX/numeric answer, unit, and (for a minority of rows) a worked solution - drawn from ten Physics, Chemistry, and Math textbooks; the entire repository is one benchmark split misleadingly named `train`.

**xw27/scibench** is the Hugging Face release of the SciBench benchmark introduced in "SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models" [1], built by curating collegiate-level problems from widely used Physics, Chemistry, and Mathematics textbooks to test multi-step reasoning, domain knowledge, and numeric computation beyond high-school-level benchmarks [1]. It lives at https://huggingface.co/datasets/xw27/scibench . **Despite the single split being named `train`, every row here is SciBench's evaluation data: this repository has no separate test split, so any row used to train a model contaminates that model's own SciBench score, and the same problems should be excluded from any other training corpus before scoring on SciBench.**

**Use it for**: held-out evaluation of scientific problem-solving (free-response reasoning against a LaTeX/numeric answer), never for SFT, DPO, or reward-model training - no training method card applies to this data; it belongs only in an evaluation harness that checks a model's generated answer against `answer_latex`/`answer_number`.

**Licence**: the raw README places `license: mit` in a YAML block at the very end of the file rather than as leading front matter, so the Hub API returns no `cardData` and no `license:` tag; the only licence signal on the repo is that trailing MIT line [2][3].

**Shape**: 692 rows, one config (`default`), one split (`train`), 8 string columns [4][5].

**Hold out**: all 692 rows - this is the benchmark itself, not training data; see the restriction above.

**Origin**: built by the SciBench authors (Wang, Hu, Lu, Zhu, Zhang, Subramaniam, Loomba, Zhang, Sun, Wang); problems and answers are human-curated from published textbooks, with no model-generated content [1][2]. As of the check date, the Hub API reports 5,746 downloads, 22,707 all-time downloads, and 27 likes [3].

**Trained-on-by**: none found - no source states a model was trained on this repository; it is documented only as an evaluation benchmark [1][2].

**Introduced by**: [1].

## Shape

Rows and columns, read live from the datasets-server (all 692 rows, offsets 0-691 in 20-row pages, so every served row was read, not sampled) [4][6]:

| split | rows |
| --- | --- |
| `train` | 692 |

One config, `default`, eight string columns [5]: `solution`, `problem_text`, `answer_latex`, `comment`, `answer_number`, `problemid`, `unit`, `source`.

Sizes (datasets-server `/size`) [4]: 444,491 bytes of original JSON download, 299,044 bytes as Parquet, 356,490 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

The `source` column names one of ten textbook acronyms. Reading every one of the 692 served rows (not a sample) gives these per-source counts, set beside the paper's own per-textbook totals and "% Solutions" figures from its Table 2 [1]; the paper's percentage times its total reproduces this card's own solved-row count almost exactly for every source, so the two are the same underlying subset even though the row totals differ:

| `source` | textbook (paper) | paper total problems | paper % with solution | rows served here | rows here with non-empty `solution` |
| --- | --- | --- | --- | --- | --- |
| `atkins` | Physical Chemistry (Atkins et al., 2014a) | 126 | 12.7% | 121 | 16 |
| `calculus` | Calculus: Early Transcendentals (Stewart et al., 2012) | 61 | 16.4% | 52 | 10 |
| `chemmc` | Quantum Chemistry (McQuarrie, 2008) | 48 | 18.8% | 47 | 9 |
| `class` | Classical Dynamics of Particles and Systems (Thornton & Marion, 2021) | 58 | 12.1% | 63 | 7 |
| `diff` | Elementary Differential Equations and Boundary Value Problems (Boyce et al., 2021) | 56 | 8.9% | 55 | 5 |
| `fund` | Fundamentals of Physics (Halliday et al., 2013) | 153 | 6.5% | 81 | 10 |
| `matter` | Physical Chemistry, Quanta, Matter, and Change (Atkins et al., 2014b) | 61 | 16.4% | 57 | 10 |
| `quan` | Quantum Chemistry (Levine et al., 2009) | 42 | 19.0% | 41 | 8 |
| `stat` | Probability and Statistical Inference (Hogg et al., 1977) | 100 | 20.0% | 92 | 20 |
| `thermo` | Statistical Thermodynamics (Engel & Reid, 2010) | 84 | 20.2% | 83 | 17 |
| total | - | 789 | - | 692 | 112 |

The paper's own totals sum to 789 problems, of which it separately states 94 form a multimodal (image-containing) subset and 103 form a distinct closed exam-question set [1]; this repository's 692 rows are neither of those, and no source states exactly which 97 of the 789 textbook problems are missing from it. One source is anomalous: `class` has 63 rows served here against 58 in the paper's table, five more than the paper's own total for that textbook, and no source explains the discrepancy.

## Quality

- The paper reports its best LLM configuration (chain-of-thought prompting plus external tools) scoring only 43.22% on this textual dataset, which the authors offer as evidence the benchmark is hard enough to separate model capability [1]. No score is reported specifically against the 692-row Hub release as opposed to the paper's own textual set, so this number should be read as characterizing the same problem pool, not a rerun on this exact file.
- The `solution` column is empty for the large majority of rows and holds a worked derivation for the rest; reading all 692 rows gives 112 rows (16.2%) with a non-empty `solution`, matching the paper's per-textbook "% Solutions" figures row for row as shown in the Shape table [1][4][6].
- No source states an annotator-agreement figure, a measured duplicate rate, or a measured contamination rate for this release; none is invented here.
- The paper states the problems were hand-extracted from PDF textbooks and re-typeset in LaTeX, and that the authors did this specifically to reduce the chance the problems already sit in an LLM's pretraining data [1] - a claim about the paper's own construction process, not a measured leakage rate for this specific Hub copy.

## Load it

Load the single split and pin the revision this card's numbers were read at (the Hub API's `sha`, matching the commit `download_checksums` are keyed to; the repo was last modified 2024-05-06) [3][5]:

```python
import datasets

REV = "93931252bc1b71d495e67390235940643d926958"
bench = datasets.load_dataset("xw27/scibench", revision=REV, split="train")  # 692 rows - eval only, hold out entirely
```

**Trap**: the split is named `train` purely because the Hub's default JSON builder calls its only split `train`; nothing about the name means the rows are meant for training. There is no separate `test` split to hold out from - the whole 692-row split is the benchmark, and it must be excluded from any SFT/RL training corpus rather than partially held out from itself [2][6].

## Neighbors

`limhyeonseok/scibench` is a translated re-release of exactly this repository: its card declares per-language configs (`es`, `ko`, `sw`, and others) each holding 692 examples, and each config's feature list repeats this repository's eight columns (`solution`, `problem_text`, `answer_latex`, `comment`, `answer_number`, `problemid`, `unit`, `source`) verbatim alongside added translation-pipeline fields (`target_lang_code`, `raw_response_text`, `is_valid_translation`, `question`, ...) [11]. Prefer this original English release for anything not specifically needing a translated prompt.

`jinulee-v/scibench` was found by the same name search but its card describes an unrelated shape - a single `default` config of 493 rows with only `id`, `question`, and `answer` columns - and states no link to this release, so it is not treated as a sibling [12].

The paper's own companion subsets - the 94-problem multimodal set and the 103-problem closed exam set - are distributed only through the authors' GitHub repository (`dataset/img` and related folders), not as separate Hugging Face datasets; the project website's only Hub link is to this repository [1][7][8][9].

## A row

One config and one split, so one shape covers the whole repository. From `config="default"`, `split="train"`, `row_idx=0` (`source="atkins"`, a row with an empty `solution`, the majority shape) [6]:

```json
{
  "solution": "",
  "problem_text": "Suppose that $10.0 \\mathrm{~mol} \\mathrm{C}_2 \\mathrm{H}_6(\\mathrm{~g})$ is confined to $4.860 \\mathrm{dm}^3$ at $27^{\\circ} \\mathrm{C}$. Predict the pressure exerted by the ethane from the perfect gas.",
  "answer_latex": " 50.7",
  "comment": " ",
  "answer_number": "50.7",
  "problemid": " e1.17(a)(a)",
  "unit": "$\\mathrm{atm}$ ",
  "source": "atkins"
}
```

For comparison, the minority shape (same schema, `solution` populated) at `row_idx=600` (`source="stat"`) [10]:

```json
{
  "solution": " By the multiplication principle, there are\n$(2)(2)(3)(2)(4)(7)(4)=2688$\ndifferent combinations.\n",
  "problem_text": "A certain food service gives the following choices for dinner: $E_1$, soup or tomato 1.2-2 juice; $E_2$, steak or shrimp; $E_3$, French fried potatoes, mashed potatoes, or a baked potato; $E_4$, corn or peas; $E_5$, jello, tossed salad, cottage cheese, or coleslaw; $E_6$, cake, cookies, pudding, brownie, vanilla ice cream, chocolate ice cream, or orange sherbet; $E_7$, coffee, tea, milk, or punch. How many different dinner selections are possible if one of the listed choices is made for each of $E_1, E_2, \\ldots$, and $E_7$ ?",
  "answer_latex": " 2688",
  "comment": " ",
  "answer_number": "2688",
  "problemid": "Example 1.2.2",
  "unit": " ",
  "source": "stat"
}
```

The schema is identical to `row_idx=0` above; only the content of `solution` differs.

## Where it came from

Built by the SciBench authors by manually selecting problems from ten published Physics, Chemistry, and Mathematics textbooks and formatting them from PDF into LaTeX, aiming to minimize the chance the problems already appear verbatim in LLM pretraining data [1]. Problem text, LaTeX and numeric answers, and units are transcribed from the textbooks; the `solution` field, populated for a minority of problems in each textbook, is a worked derivation the authors kept for detailed error analysis [1]. No model-generated content is involved in constructing the problems or answers; the human curation and transcription is the sole generation process [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hub repositories are mutable, which is why Load it pins the revision.

[1] Wang, Hu, Lu, Zhu, Zhang, Subramaniam, Loomba, Zhang, Sun, Wang, "SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models", ICML 2024. https://arxiv.org/abs/2307.10635 - abstract, Table 2 per-textbook counts and % Solutions, multimodal/closed-set sizes, construction process. Abstract read from the live abs page; Table 2 and body text read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2307.10635. Fetched 2026-08-11.

[2] xw27/scibench dataset card (README). https://huggingface.co/datasets/xw27/scibench/raw/main/README.md - description, trailing `license: mit` block. Fetched 2026-08-11.

[3] Hugging Face Hub API record for xw27/scibench. https://huggingface.co/api/datasets/xw27/scibench?full=true (no `cardData` field) and https://huggingface.co/api/datasets/xw27/scibench?expand[]=downloadsAllTime - sha, downloads, downloadsAllTime, likes, lastModified. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=xw27%2Fscibench Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=xw27%2Fscibench - column dtypes, per-file `download_checksums` keyed to commit `93931252bc1b71d495e67390235940643d926958`. Fetched 2026-08-11.

[6] datasets-server rows endpoint, called across offsets 0, 20, 40, ..., 680 with length 20 (and a first-rows call at offset 0), covering all 692 rows of `config=default`, `split=train`. https://datasets-server.huggingface.co/rows?dataset=xw27%2Fscibench&config=default&split=train Fetched 2026-08-11.

[7] scibench-ucla.github.io project website - the only Hugging Face link on the page is to this repository. https://scibench-ucla.github.io Fetched 2026-08-11.

[8] github.com/mandyyyyii/scibench repository listing, via the GitHub contents API. https://api.github.com/repos/mandyyyyii/scibench/contents and https://api.github.com/repos/mandyyyyii/scibench/contents/dataset - top-level and `dataset/` folder contents (`dataset/img`, `dataset/original`). Fetched 2026-08-11.

[9] xw27/scibench repository, older README draft retained in the tree. https://huggingface.co/datasets/xw27/scibench/raw/main/.ipynb_checkpoints/README-checkpoint.md - states "695 problems", a figure this card does not adopt since it does not match the 692 rows actually served. Fetched 2026-08-11.

[10] datasets-server rows endpoint, single-row call. https://datasets-server.huggingface.co/rows?dataset=xw27%2Fscibench&config=default&split=train&offset=600&length=1 Fetched 2026-08-11.

[11] limhyeonseok/scibench dataset card (README). https://huggingface.co/datasets/limhyeonseok/scibench/raw/main/README.md - per-language config feature lists and 692-example counts. Fetched 2026-08-11.

[12] jinulee-v/scibench dataset card (README). https://huggingface.co/datasets/jinulee-v/scibench/raw/main/README.md - `default` config feature list and 493-example count. Fetched 2026-08-11.

[13] The corpus screening row for `xw27/scibench`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable only as a held-out evaluation benchmark, never as training data. The card's own restriction rests on facts established above: the repository has exactly one split, named `train`, containing what the origin paper describes as its evaluation problem set, with no separate held-out test split of its own [1][2][6]. The screening row's flag calls this out directly as a naming trap [13].

### The screening row

The row's own note [13]: "SciBench college textbook science problems; its one split is named `train` but it is the benchmark." Its flag: "trap: the SciBench benchmark with its only split named train."
