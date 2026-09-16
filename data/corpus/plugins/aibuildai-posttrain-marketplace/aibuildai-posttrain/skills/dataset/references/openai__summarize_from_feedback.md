# openai/summarize_from_feedback

193,841 rows of OpenAI's human-feedback data for summarization: pairwise preference comparisons over model-written summaries, and separate Likert-scale quality ratings, both built on the Reddit TL;DR corpus plus CNN/Daily Mail articles.

**openai/summarize_from_feedback** is the human-feedback release behind "Learning to summarize from human feedback" [1]: OpenAI crowdworkers compared pairs of machine-written summaries of Reddit posts and news articles, and the resulting preference labels trained the reward model used to RLHF-tune the paper's summarization policy [2]. The repository holds two configs with different row shapes: `comparisons` (pairwise preference choices, split `train`/`validation`) and `axis` (single-summary Likert ratings on four quality axes, split `test`/`validation`) [2]. **The `axis` config carries no `train` split - only `test` and `validation` - and the GitHub release lists it under the paper's evaluation data on TL;DR and CNN/Daily Mail rather than among the comparisons used to train the reward model, so hold it out of any training run** [2][3]. It lives at https://huggingface.co/datasets/openai/summarize_from_feedback .

**Use it for**: preference-pair training - the `comparisons` config feeds a reward model or a preference-optimization method, never the `axis` ratings. Each `comparisons` row nests two candidate summaries under `summaries[0]`/`summaries[1]` plus an integer `choice` (0 or 1) picking the preferred one, so building a standard chosen/rejected preference pair means indexing `summaries` by `choice` and constructing a prompt from `info.post`/`info.article` - it does not arrive as TRL's ready-made implicit- or explicit-prompt preference format out of the box [4]. See the SFT method card for the separate TL;DR summarization SFT data.

**Licence**: not stated - the repo's `cardData` carries only `pretty_name`, no `license` key, and no `license:` tag appears among the repo tags [5]. The one catch: the GitHub release notes the underlying Reddit TL;DR corpus (Syed et al. 2018) that these prompts are drawn from is licensed CC BY 4.0, but that licence covers the source posts, not a licence term stated by this repository itself [3].

**Shape**: 193,841 rows across two configs - `axis` (`test` 6,312 + `validation` 8,585 = 14,897 rows, 5 columns) and `comparisons` (`train` 92,858 + `validation` 86,086 = 178,944 rows, 7 columns) [6][7].

**Hold out**: the entire `axis` config (14,897 rows: 6,312 `test` + 8,585 `validation`) - it is evaluation-only per the card and the GitHub release [2][3]. Within `comparisons/validation` (86,086 rows), rows also carry a `split` field valued `valid1` or `valid2`; the GitHub release states that posts marked `valid1` were used for model selection during training, while `valid2` was reserved for final evaluation, so a clean held-out eval slice means filtering `split=="valid2"`, not the whole 86,086-row split [3].

**Origin**: built and released by OpenAI; the two candidate summaries in `comparisons` are OpenAI-model generations (various supervised and PPO policies), and the preference/rating labels are human crowdworker judgments [2][3]. Hub API at the check date: `downloads` 2,478, `downloadsAllTime` 132,868, `likes` 220 [5].

**Trained-on-by**: the origin paper's own reward model and PPO-tuned summarization policy - this release is that paper's feedback data [1][2]. The DPO paper's TL;DR summarization experiment states it uses "the Reddit TL;DR summarization dataset along with human preferences gathered by Stiennon et al. 2022," and starts from a checkpoint hosted as `CarperAI/openai_summarize_tldr_sft`, trained with the TRLX RLHF framework [8]. TRLX's own summarization example pipeline trains a reward-model checkpoint (`CarperAI/openai_summarize_tldr_rm_checkpoint`) and reports ROUGE/reward scores for SFT and PPO models on the TL;DR test set [9]; the pipeline's reward-model training script defaults to loading `CarperAI/openai_summarize_comparisons`, the reformatted mirror of this release's `comparisons` config, as its training data [10].

**Introduced by**: [1] (Stiennon et al.).

## Shape

Rows and columns per config, from the datasets-server size and info endpoints [6][7]:

| config | split | rows | columns |
| --- | --- | --- | --- |
| `axis` | `test` | 6,312 | 5 |
| `axis` | `validation` | 8,585 | 5 |
| `comparisons` | `train` | 92,858 | 7 |
| `comparisons` | `validation` | 86,086 | 7 |
| total | | 193,841 | |

`axis` columns: `info` (struct: `id`, `post`, `title`, `subreddit`, `site`, `article`), `summary` (struct: `text`, `policy`, `note`, `axes` - a struct of `overall`/`accuracy`/`coverage`/`coherence` int scores plus `compatible` bool), `worker`, `batch`, `split` [7]. `comparisons` columns: `info` (same six-field struct), `summaries` (a list of two structs, each `text`/`policy`/`note`), `choice` (int), `worker`, `batch`, `split`, `extra` (struct: `confidence`) [7]. No source states token or sequence-length statistics for this release.

Sizes from the datasets-server size endpoint [6]: `axis` is 45,339,339 bytes of original JSON download / 42,239,761 bytes decoded in memory; `comparisons` is 374,387,954 bytes original / 343,118,863 bytes decoded; the whole repository totals 419,727,293 bytes of original download, 48,547,221 bytes as Parquet, and 385,358,624 bytes decoded in memory.

The datasets-server first-rows endpoint returns rows from offset 0 up to a byte-size cap, not a fixed row count: 51 rows for `axis/test`, 100 for `axis/validation`, 91 for `comparisons/train`, and 90 for `comparisons/validation` [11][12][13][14]. Within those samples, every `axis/test` row is a news-article summary (`info.site` set to `cnn` or `dailymail`, `info.subreddit` null), while every row in `axis/validation`, `comparisons/train`, and `comparisons/validation` is a Reddit post (`info.subreddit` set, `info.site` null) - a pattern read only in these offset-0 samples per split, not confirmed across the full splits.

## Quality

- Each `summaries[*].note` (or `summary.note` in `axis`) is a crowdworker's own written rationale before seeing the full post; the GitHub release states these notes "contain the naive interpretation notes written by the worker before seeing the post (but possibly edited afterwards)" and may be null [3].
- On a subset of comparison tasks, the origin paper reports that labelers agreed with the paper's researchers 77% ± 2% of the time, while the researchers agreed with each other 73% ± 4% of the time, citing this as evidence of high labeler-researcher agreement achieved through their quality-control procedure [1].
- The GitHub release's `split` field documentation is a quality-relevant caveat for `comparisons`: `valid1` rows were used for model selection during training, so only `valid2` rows represent a fully held-out evaluation slice [3]. Sampling 500 rows of `comparisons/validation` at five offsets (0, 20,000, 40,000, 60,000, 80,000 out of 86,086, 100 rows each, via the datasets-server rows endpoint) found 259 `valid1` rows and 241 `valid2` rows (about 52%/48%), with both values mixed at every offset except 0 and 80,000, where the 100-row sample was entirely `valid1` [15]. The `split` column itself is the join key for filtering to `valid2` - no external fetch is needed.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-01-03) [5]:

```python
import datasets

REV = "b8f7d168b6f4e95b2a92e84768bd6c955bed2f29"  # main at the check date
comparisons_train = datasets.load_dataset("openai/summarize_from_feedback", "comparisons", revision=REV, split="train")            # 92,858 rows
comparisons_val = datasets.load_dataset("openai/summarize_from_feedback", "comparisons", revision=REV, split="validation")          # 86,086 rows - filter split=="valid2" for a clean eval slice
axis_test = datasets.load_dataset("openai/summarize_from_feedback", "axis", revision=REV, split="test")                             # 6,312 rows - evaluation only, do not train on it
axis_validation = datasets.load_dataset("openai/summarize_from_feedback", "axis", revision=REV, split="validation")                 # 8,585 rows - evaluation only, do not train on it
```

**Trap**: the two configs have incompatible schemas - `comparisons` nests two summaries under `summaries`/`choice`, `axis` carries one summary under `summary`/`axes` - so a collator written for one config will not run on the other; always load `comparisons` and `axis` separately, never concatenated. A second trap is the `comparisons/validation` split itself: loading it whole silently mixes `valid1` (used for model selection during training) with `valid2` (final eval) rows, per the row-level finding above [3][15].

## Neighbors

- `HuggingFaceH4/summarize-from-feedback` - the `comparisons` config only, column-renamed (`meta`/`responses`/`label` in place of `info`/`summaries`/`choice`) but row-for-row identical in count: 92,858 train / 86,086 validation, matching this release exactly [16].
- `CarperAI/openai_summarize_comparisons` - reformatted into explicit `prompt`/`chosen`/`rejected` string columns, ready for TRL/TRLX-style reward-model training. Its total row count (259,960 across `train` 92,534 / `test` 83,629 / `valid1` 33,082 / `valid2` 50,715) is larger than this release's 178,944 `comparisons` rows and its own card does not explain the difference, so treat it as a reformatted near-copy rather than an exact mirror [17].
- `CarperAI/openai_summarize_tldr` - a different task shape entirely: SFT pairs of `prompt`/`label` (post plus its written TL;DR), 129,722 rows (116,722 train / 6,447 valid / 6,553 test), built for supervised fine-tuning on TL;DR rather than preference training. See the SFT method card for this shape [18].
- This corpus prefers the original `openai/summarize_from_feedback` `comparisons` config for preference training, since the reformatted neighbors either duplicate it under new column names or diverge from its row counts without explanation.

## A row

Two configs, two schemas - one row from each, both fetched live via the datasets-server first-rows endpoint [11][12].

From `config="axis"`, `split="test"`, `row_idx=0` (news-article domain), article truncated:

```json
{
  "info": {
    "id": "167f80cc6634b166a699d182e25c81a2349d82d2",
    "post": null,
    "title": "Newcastle United midfielder Moussa Sissoko faces disciplinary action from the club after dangerous tackle on Lucas Leiva",
    "subreddit": null,
    "site": "dailymail",
    "article": "Newcastle stand-in skipper Moussa Sissoko is facing disciplinary action after he was sent off following a reckless challenge on Liverpool midfielder Lucas Leiva during Monday's 2-0 defeat at Anfield. [...] Newcastle's players appear dejected as Joe Allen celebrates scoring Liverpool's second goal at Anfield"
  },
  "summary": {
    "text": "Moussa Sissoko was sent off against Liverpool on Monday night.. John Carver felt that Sissoko's second booking was worthy of a red card.. Midfielder could be punished by his club on top of a two-game ban.. Carver admits he is only concerned with results and not performances.. Newcastle are 13th in the table, nine points off the relegation zone.",
    "policy": "ref",
    "note": "Misleading: \"Carver admits he is only concerned with results and not performances\" understood as if critics of monday's match but it's said for the following matches.\n\n13th??\n\nDoesnt properly address the teams, the match, the result, 2nd yellow card and therefore sent off, etc.",
    "axes": {
      "overall": 3,
      "accuracy": 5,
      "coverage": 4,
      "coherence": 2,
      "compatible": null
    }
  },
  "worker": "qo6WIyEh27cwAjWpA3Q60J7NaDxzQJ",
  "batch": "cnndm1",
  "split": "test"
}
```

From `config="comparisons"`, `split="train"`, `row_idx=0` (Reddit domain):

```json
{
  "info": {
    "id": "t3_34xale",
    "post": "My boyfriend and I are long distance. We have a trip planned this summer which involves me going over to him in the USA. [...] I feel like I have done everything I can to make her feel comfortable with this trip and she is just trying to sabotage it. Thoughts??",
    "title": "Mother [51] not speaking to me [21] because of a trip I am planning",
    "subreddit": "relationships",
    "site": null,
    "article": null
  },
  "summaries": [
    {"text": " Mum is mad at me for not flying on my own trip to meet my boyfriend.", "policy": "sup1", "note": null},
    {"text": " I have made sure my mother is comfortable with my boyfriend travelling on a trip and now my mother is mad because I booked it.", "policy": "sup1", "note": null}
  ],
  "choice": 1,
  "worker": "qo6WIyEh27cwAjWpA3Q60J7NaDxzQJ",
  "batch": "batch3",
  "split": "train",
  "extra": {"confidence": null}
}
```

## Where it came from

Built and released by OpenAI alongside "Learning to summarize from human feedback" [1]. The GitHub release describes the `comparisons` config as crowdworkers comparing pairs of summaries written by different OpenAI policies (supervised baselines and PPO-tuned models at various stages, plus a reference summary) for the same Reddit post [3]. The GitHub release's own summary line describes the release as a whole as containing 64,832 summary comparisons on the TL;DR dataset, plus the paper's evaluation data on TL;DR and CNN/Daily Mail - a figure roughly a third of the 178,944 rows this card measures directly in the Hub `comparisons` config today; the release does not reconcile the two counts, so treat 64,832 as the paper-era description and 178,944 as the number this card's Shape table and Load it commands are pinned to [3][6]. The `axis` config carries the release's Likert-scale ratings on both TL;DR and CNN/Daily Mail [3]. The underlying Reddit posts come from OpenAI's filtered version of the TL;DR dataset originally introduced by Syed et al. 2018, and the CNN/Daily Mail articles used in `axis/test` are a separate news-summarization evaluation domain [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Stiennon et al., "Learning to summarize from human feedback", NeurIPS 2020. https://arxiv.org/abs/2009.01325 - the origin paper; current title read from the live abs page, labeler-researcher agreement figure read from the full text via ar5iv (https://ar5iv.labs.arxiv.org/html/2009.01325). Fetched 2026-08-11.

[2] openai/summarize_from_feedback dataset card (README). https://huggingface.co/datasets/openai/summarize_from_feedback/raw/main/README.md - config descriptions, split names, source datasets, GitHub link. Fetched 2026-08-11.

[3] The GitHub release notes for the human feedback data. https://raw.githubusercontent.com/openai/summarize-from-feedback/master/README.md - `comparisons`/`axis_evals` descriptions, `split` field semantics (`train`/`valid1`/`valid2`), worker note description, TL;DR corpus provenance and its CC BY 4.0 licence. Fetched 2026-08-11.

[4] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats - implicit/explicit-prompt preference format definitions. A `main` build, unpinned and mutable. Fetched 2026-08-11.

[5] Hugging Face Hub API record for openai/summarize_from_feedback. https://huggingface.co/api/datasets/openai/summarize_from_feedback?full=true - licence (absent), gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=openai%2Fsummarize_from_feedback Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=openai%2Fsummarize_from_feedback Fetched 2026-08-11.

[8] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - TL;DR summarization experiment setup, naming this dataset's authors and the `CarperAI/openai_summarize_tldr_sft` checkpoint. Read as the arXiv HTML full text via ar5iv (https://ar5iv.labs.arxiv.org/html/2305.18290). Fetched 2026-08-11.

[9] TRLX summarization example README. https://raw.githubusercontent.com/CarperAI/trlx/main/examples/summarize_rlhf/README.md - names the reward-model and PPO checkpoints and gives their ROUGE/reward evaluation on the TL;DR test set; does not itself name a training-data path. Fetched 2026-08-11.

[10] TRLX reward-model training script. https://raw.githubusercontent.com/CarperAI/trlx/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py - `create_comparison_dataset` and the `data_path` default both point at `CarperAI/openai_summarize_comparisons`. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint, `axis`/`test`. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fsummarize_from_feedback&config=axis&split=test Fetched 2026-08-11.

[12] datasets-server first-rows endpoint, `comparisons`/`train`. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fsummarize_from_feedback&config=comparisons&split=train Fetched 2026-08-11.

[13] datasets-server first-rows endpoint, `axis`/`validation`. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fsummarize_from_feedback&config=axis&split=validation Fetched 2026-08-11.

[14] datasets-server first-rows endpoint, `comparisons`/`validation`. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fsummarize_from_feedback&config=comparisons&split=validation Fetched 2026-08-11.

[15] datasets-server rows endpoint, `comparisons`/`validation`, sampled at offsets 0, 20000, 40000, 60000, 80000 (length 100 each). https://datasets-server.huggingface.co/rows?dataset=openai%2Fsummarize_from_feedback&config=comparisons&split=validation&offset=<n>&length=100 Fetched 2026-08-11.

[16] HuggingFaceH4/summarize-from-feedback dataset card and datasets-server size endpoint. https://huggingface.co/datasets/HuggingFaceH4/summarize-from-feedback/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=HuggingFaceH4%2Fsummarize-from-feedback Fetched 2026-08-11.

[17] CarperAI/openai_summarize_comparisons dataset card and first-rows endpoint. https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons/raw/main/README.md and https://datasets-server.huggingface.co/first-rows?dataset=CarperAI%2Fopenai_summarize_comparisons&config=default&split=train Fetched 2026-08-11.

[18] CarperAI/openai_summarize_tldr dataset card. https://huggingface.co/datasets/CarperAI/openai_summarize_tldr/raw/main/README.md Fetched 2026-08-11.

[19] The corpus screening row for `openai/summarize_from_feedback`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, split by config: `comparisons` (178,944 rows) is preference-pair training data; `axis` (14,897 rows) is evaluation-only and must be held out of training. This rests on facts already established above - the GitHub release describes `axis_evals` as evaluation data on both TL;DR and CNN/DM [3], and the screening row's note draws the same line [19].

### The screening row

The row's own note [19]: "`comparisons` (train/validation) are human preference labels over model-written summaries; the `axis` config is test/validation human quality ratings and is evaluation only." The row carries no flag.
