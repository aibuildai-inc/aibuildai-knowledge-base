# deepmind/code_contests

13,610 competitive-programming problems - natural-language descriptions with paired public/private/generated test cases, human-written correct and incorrect solutions in up to four languages, and per-problem source/difficulty metadata - DeepMind's training and evaluation corpus for AlphaCode.

**deepmind/code_contests** is DeepMind's CodeContests dataset, built by scraping competitive-programming problems from Aizu, AtCoder, CodeChef, Codeforces and HackerEarth - Aizu and AtCoder via the CodeNet corpus [1], CodeChef and HackerEarth via the Description2Code corpus, and Codeforces via both Description2Code and direct scraping [3] - introduced in "Competition-Level Code Generation with AlphaCode" [2] as the corpus AlphaCode was fine-tuned and evaluated on. Each row is one contest problem: a natural-language statement, paired input/output test cases, and lists of human solutions and incorrect human solutions, framed by the dataset card as a translation task from a natural-language description to a program that satisfies the tests [3]. **The `test` split is the CodeContests competitive-programming benchmark itself, per the corpus screening note for this dataset [4]: hold it out of any training run and reserve it for evaluation.** It lives at https://huggingface.co/datasets/deepmind/code_contests .

**Use it for**: reasoning-trace SFT on human solution code, or execution-verifiable RL, using the `train` (and, per the screening note, `valid` [4]) splits - hold out `test` (see above). Each row bundles a `description` string with nested `solutions`/`incorrect_solutions` lists (language + code) and `public_tests`/`private_tests`/`generated_tests` (paired `input`/`output` strings usable as an execution reward), not preference pairs; building single-turn SFT examples requires exploding the `(language, solution)` list per problem into individual training rows. Maps to the SFT method card for the solutions, or a verifiable-reward RL method card using the bundled tests as the reward signal.

**Licence**: `cc-by-4.0` for the dataset as packaged [5][3], ungated [5]. The one catch: the card itself notes that the Description2Code-sourced material is separately licensed MIT (copyright unspecified) and the CodeNet-sourced material separately licensed Apache 2.0 (copyright unspecified) [3] - the CC-BY-4.0 label covers DeepMind's packaging, not a claim that it supersedes those upstream terms.

**Shape**: one config, three splits. The repository's own build metadata (`dataset_infos.json`) declares `train` 13,328 / `valid` 117 / `test` 165 rows, 13,610 total [6]; the Hub's live datasets-server API currently reports only `train` 3,762 / `valid` 117 / `test` 165, 4,044 total, flagged `"partial": true` [7] - see Shape below.

**Hold out**: `test` (165 rows in both the build metadata and the live partial API, so this count is not in dispute) - the CodeContests benchmark; do not train on it. `valid` (117 rows, likewise undisputed) is not flagged as a benchmark by the screening note and may be used for training [4].

**Origin**: built and released by DeepMind (Yujia Li, David Choi and co-authors listed as dataset curators) [3]; content is human-authored - problem statements and solutions scraped from the source judges/corpora, not model-generated [3]. Hub API as of the check date: 50,563 recent downloads, 7,608,835 all-time downloads, 229 likes [5].

**Trained-on-by**: AlphaCode itself was fine-tuned on this dataset, per the origin paper [2]. AlphaCode 2 was fine-tuned on what its technical report calls an "updated version of the CodeContests dataset" ("CodeContests v2"), described as containing about 15,000 problems and 30 million human code samples - a larger, revised corpus than this exact release, not confirmed to be this Hub repository verbatim [8]. No other adopter was found in the sources checked for this card.

**Introduced by**: [2] (Li et al., "Competition-Level Code Generation with AlphaCode").

## Shape

Row counts, two disagreeing sources:

| split | `dataset_infos.json` (build, in-repo) [6] | datasets-server `/size` (live, `partial: true`) [7] |
| --- | --- | --- |
| `train` | 13,328 | 3,762 |
| `valid` | 117 | 117 |
| `test` | 165 | 165 |
| total | 13,610 | 4,044 |

`valid` and `test` agree exactly between the two sources; only `train` is undercounted by the live API. The Hub's own auto-generated `size_categories` tag on this repo reads `1K<n<10K` [5], consistent with the live partial count, while the README's own YAML front matter declares `size_categories: 10K<n<100K` [3], consistent with the full 13,610-row build - the tag and the card front matter disagree for the same reason.

Twenty columns, one schema shared by all three splits (datasets-server `/info`) [9]:

| column | dtype |
| --- | --- |
| `name` | string |
| `description` | string |
| `public_tests` | list\<struct\<input: string, output: string\>\> |
| `private_tests` | list\<struct\<input: string, output: string\>\> |
| `generated_tests` | list\<struct\<input: string, output: string\>\> |
| `source` | class_label (`UNKNOWN_SOURCE`, `CODECHEF`, `CODEFORCES`, `HACKEREARTH`, `CODEJAM`, `ATCODER`, `AIZU`) |
| `difficulty` | class_label (`UNKNOWN_DIFFICULTY`, `EASY`...`HARDEST`, `EXTERNAL`, `A`...`V`) |
| `solutions` | list\<struct\<language: class_label, solution: string\>\> |
| `incorrect_solutions` | list\<struct\<language: class_label, solution: string\>\> |
| `cf_contest_id` | int64 |
| `cf_index` | string |
| `cf_points` | float32 |
| `cf_rating` | int32 |
| `cf_tags` | list\<string\> |
| `is_description_translated` | bool |
| `untranslated_description` | string |
| `time_limit` | struct\<seconds: int64, nanos: int64\> |
| `memory_limit_bytes` | int64 |
| `input_file` | string |
| `output_file` | string |

`solutions.language` and `incorrect_solutions.language` share one class label set: `UNKNOWN_LANGUAGE`, `PYTHON` (Python 2), `CPP`, `PYTHON3`, `JAVA` [3].

Byte sizes: the build metadata records 7,624,659,530 bytes of Parquet download and 19,397,165,916 bytes decoded in memory for the full 13,610-row dataset [6]; the live, partial datasets-server scan instead reports 2,220,949,795 Parquet bytes and 5,686,273,386 in-memory bytes, covering only the 4,044 rows it actually scanned [7]. No source states sequence-length or token statistics for this dataset.

## Quality

- Solutions were scraped alongside their problem descriptions from the source judges, and the README lists the source-data creators as the annotators, i.e. there is no separate human-annotation step beyond scraping [3].
- No source states a measured contamination rate, duplicate rate, or solution-correctness verification rate for this dataset; none is invented here.
- The README recommends `cf_rating` over the categorical `difficulty` field for Codeforces problems, since difficulty gradings are not comparable across the five source sites [3].
- Of the first 10 rows read from `train` at offset 0, none had `is_description_translated` set to true [10]. Six were CodeChef problems (`source`=1) with between 1 and 47 correct solutions each and no `incorrect_solutions`; the other four were Codeforces problems (`source`=2), and each of those four did carry a non-empty `incorrect_solutions` list - the served field is truncated to 100 characters by the API but that truncated fragment alone already contains dozens of comma-separated language codes for each of the four [10]. This describes only those 10 rows, not the split as a whole.
- Fine-tuning on this dataset was, in the origin paper's own words, "critical for performance" relative to a GitHub-pretrained-only model, per the curation rationale quoted in the dataset card [3]. The paper's headline result for the resulting AlphaCode system is an average top-54.3% ranking in simulated Codeforces competitions with more than 5,000 participants [2].

## Load it

Pin the revision this card's numbers were read at (matches the shortlist's pinned commit and the Hub API's current `sha` for `main`; the repo was last modified 2023-06-11) [5]:

```python
import datasets

REV = "802411c3010cb00d1b05bad57ca77365a3c699d6"  # main at the check date
train = datasets.load_dataset("deepmind/code_contests", revision=REV, split="train")  # 13,328 rows
valid = datasets.load_dataset("deepmind/code_contests", revision=REV, split="valid")  # 117 rows
test = datasets.load_dataset("deepmind/code_contests", revision=REV, split="test")    # 165 rows - hold out
```

**Trap**: `load_dataset` reads the full Parquet shards and returns all 13,328 `train` rows; the Hub's dataset viewer and the datasets-server `/size` and `/info` APIs are marked `"partial": true` for this repo and currently report only 3,762 `train` rows [7] - an agent that sizes its training run off the live viewer or API instead of actually loading the split will underestimate `train` by roughly 3.5x. The full in-memory size is about 19.4 GB [6], so loading `train` fully needs enough RAM or disk to stream it rather than materializing it all at once.

## Neighbors

- `teven/code_contests` - a flattened re-release, one row per solution instead of one row per problem, keeping only `name`/`source`/`description`/`solution`/`language`/`difficulty` and dropping all test cases and incorrect solutions; its own card describes it as the "HF-datasets version" of this dataset [11][12]. Its live `/size` reports 4,432,447 `train` rows, 32,181 `test`, 29,863 `valid` [11] - far more rows than this release because each solution is its own row, and it is missing the executable tests this card's rows carry, so it is not interchangeable with `deepmind/code_contests` for execution-reward RL.
- `BEE-spoke-data/code_contests_instruct` - an instruction-formatted, multi-config re-release (`default`, `hq`, `hq-deduped`, `hq-python`, `hq-python-deduped`, `min-cols`, and more) built from both `teven/code_contests` and this repository, per its own card [13][12]. Its `default` config's live `/size` matches `teven/code_contests` row-for-row (4,432,447 `train` / 32,181 `test` / 29,863 `valid`) [13], consistent with it being a reformatting of the flattened release rather than of this one.
- `talrid/CodeContests_valid_and_test_AlphaCodium` - a repackaging of only this dataset's `valid` and `test` splits, shipped as a single zip archive rather than Parquet (no rows are served through datasets-server) [14]; it covers the AlphaCode benchmark portion of this corpus, not additional training data, and offers nothing this card's own `valid`/`test` splits do not already have.

This corpus prefers `deepmind/code_contests` itself over these neighbors whenever the executable tests or the per-problem (rather than per-solution) shape are needed; reach for `teven/code_contests` or `BEE-spoke-data/code_contests_instruct` only when a flattened, tests-free, one-solution-per-row shape is what the trainer expects.

## A row

One schema is served across all three splits (see Shape), so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [10], with the long `generated_tests` list truncated:

```json
{
  "name": "brcktsrm",
  "description": "Problem description.\nVipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the world. ... \n\nConstraints\n\n1 <= T <= 10\n1 <= length of S <= 60\n\n\nExample\nInput:\n3\n((()))\n(())()\n()(()\n\nOutput:\nYES\nYES\nNO\n\n \n\nExplanation\nExample is self-explanatory.",
  "public_tests": {
    "input": ["3\n((()))\n(())()\n()(()"],
    "output": ["YES\nYES\nNO"]
  },
  "private_tests": {"input": [], "output": []},
  "generated_tests": {
    "input": ["3\n((()))\n(())()\n()())", "3\n((()()\n(())()\n()(()", "... [98 more generated test inputs, 100 total]"],
    "output": ["...100 paired outputs..."]
  },
  "source": 1,
  "difficulty": 6,
  "solutions": {"language": [1, 1, 1], "solution": ["for _ in range(input()):\n    try:\n        eval(raw_input())\n        print 'YES'\n ...", "...2 more..."]},
  "incorrect_solutions": {"language": [], "solution": []},
  "cf_contest_id": 0,
  "cf_index": "",
  "cf_points": 0.0,
  "cf_rating": 0,
  "cf_tags": [],
  "is_description_translated": false,
  "untranslated_description": "",
  "time_limit": null,
  "memory_limit_bytes": 0,
  "input_file": "",
  "output_file": ""
}
```

`source=1` is `CODECHEF`; `cf_contest_id`/`cf_index`/`cf_rating` are zero/blank because this problem is not from Codeforces. This row has no `private_tests` and no `incorrect_solutions` - both lists can be empty on a given row.

## Where it came from

DeepMind built and released this dataset. Problems come from five judges: Aizu and AtCoder via the CodeNet corpus [1]; CodeChef and HackerEarth via the Description2Code corpus; and Codeforces via both Description2Code and direct scraping [3]. The dataset card states that data collection and normalization are documented in Section 3.2 and Appendix B.2 of the origin paper [3][2]. Solutions - both correct and incorrect - were scraped alongside each problem's description from the same source judges; there is no separate human-labeling step [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Puri et al., "CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks", 2021. https://arxiv.org/abs/2105.12655 - current title read from the live abs page; the source of Aizu/AtCoder problems per the README. Fetched 2026-08-11.

[2] Li et al., "Competition-Level Code Generation with AlphaCode", 2022. https://arxiv.org/abs/2203.07814 - the origin paper; current title read from the live abs page; abstract states the top-54.3% average competition ranking. Fetched 2026-08-11.

[3] deepmind/code_contests dataset card (README). https://huggingface.co/datasets/deepmind/code_contests/raw/main/README.md - task framing, data fields, source-data description, curation-rationale quote, licensing section, dataset curators. Fetched 2026-08-11.

[4] The corpus screening row for `deepmind/code_contests`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[5] Hugging Face Hub API record for deepmind/code_contests. https://huggingface.co/api/datasets/deepmind/code_contests?full=true - licence tag, gate status, `sha`, `downloads`, `likes`, last-modified date, `size_categories` tag; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] dataset_infos.json in the repository tree. https://huggingface.co/datasets/deepmind/code_contests/raw/main/dataset_infos.json - build-time canonical split row counts, download size, and in-memory dataset size. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=deepmind%2Fcode_contests - returns `"partial": true`; live, unpinned, no revision parameter. Fetched 2026-08-11.

[8] "AlphaCode 2 Technical Report", AlphaCode Team, Google DeepMind, 2023-12-06. https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf - fine-tuning on an "updated version of the CodeContests dataset" ("CodeContests v2"), ~15,000 problems and 30 million human code samples. Fetched 2026-08-11.

[9] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=deepmind%2Fcode_contests Fetched 2026-08-11.

[10] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=deepmind%2Fcode_contests&config=default&split=train Fetched 2026-08-11.

[11] datasets-server size endpoint for teven/code_contests, plus its README. https://datasets-server.huggingface.co/size?dataset=teven%2Fcode_contests and https://huggingface.co/datasets/teven/code_contests/raw/main/README.md Fetched 2026-08-11.

[12] Neighbor dataset cards read for what each re-release says it did: `teven/code_contests` and `BEE-spoke-data/code_contests_instruct`. https://huggingface.co/datasets/<id>/raw/main/README.md Fetched 2026-08-11.

[13] datasets-server size endpoint for BEE-spoke-data/code_contests_instruct, plus its README. https://datasets-server.huggingface.co/size?dataset=BEE-spoke-data%2Fcode_contests_instruct and https://huggingface.co/datasets/BEE-spoke-data/code_contests_instruct/raw/main/README.md Fetched 2026-08-11.

[14] Hugging Face Hub API record for talrid/CodeContests_valid_and_test_AlphaCodium, and its live size endpoint (empty response, since the repo ships a zip archive rather than Parquet). https://huggingface.co/api/datasets/talrid/CodeContests_valid_and_test_AlphaCodium and https://datasets-server.huggingface.co/size?dataset=talrid%2FCodeContests_valid_and_test_AlphaCodium Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, with the split-level restriction the card leads with: train on `train` and `valid`, hold out `test` because it is the CodeContests benchmark. This rests on the screening row's own note [4] and on the row counts and split identity already established above (Shape, Hold out).

### The screening row

The row's own note [4]: "the AlphaCode competitive-programming corpus (problems, human submissions, generated tests); train/valid are safe, `test` is the CodeContests benchmark." The row carries no flag.
