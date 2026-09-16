# stanfordnlp/SHP

385,563 Reddit preference pairs across 18 subreddits, each pair a naturally occurring question/instruction with two top-level comments whose collective human vote order gives the preference label, split train/validation/test.

**stanfordnlp/SHP** (the Stanford Human Preferences Dataset) was built by Kawin Ethayarajh, Heidi (Chenyu) Zhang, Yizhong Wang, and Dan Jurafsky at Stanford, scraped from Reddit, and the dataset card asks users to cite "Understanding Dataset Difficulty with V-Usable Information" (ICML 2022) as the paper whose techniques created it [1][2]. It lives at https://huggingface.co/datasets/stanfordnlp/SHP . Each row pairs a Reddit post (`history`) with two top-level comments (`human_ref_A`, `human_ref_B`); the label exploits the fact that if a later comment nonetheless scored higher than an earlier one, it is inferred to be the more-preferred reply, since an earlier comment's higher score could just reflect more visibility [2]. **The card states the preference label reflects helpfulness, not harmlessness, and says SHP "is not intended for use in harm-minimization"; it also warns the more-preferred response "is not necessarily the more factual one"** [2].

**Use it for**: preference-pair training - a reward model or a preference-optimization method - not SFT; the card frames it for "training RLHF reward models and NLG evaluation models" [2]. Rows are the implicit-prompt preference format (`human_ref_A`/`human_ref_B` are the two candidate replies to the shared `history` prompt, and `labels` says which one is preferred); map `labels==1` to A-preferred, `labels==0` to B-preferred before building `chosen`/`rejected` columns. See the DPO or reward-model method card.

**Licence**: no SPDX id in `cardData`; the card's own License section says the data was scraped "in accordance with the Reddit API Terms of Use" without any direct agreement with Reddit, and states Reddit's "User Content" licence is "non-exclusive, non-transferable, non-sublicensable, and revocable" [2]. Ungated (`"gated": false`, `"private": false`) [3]. The one catch: the card explicitly disclaims responsibility for downstream use and reserves the right to modify the dataset or licence at any point [2].

**Shape**: 385,563 rows in one config (`default`), split `train` 348,718 / `validation` 18,436 / `test` 18,409, 15 columns [4][5].

**Hold out**: `validation` (18,436 rows) and `test` (18,409 rows); train on `train` (348,718). The card's own split table gives these exact counts and states splits were built by post ID, so no post appears in more than one split [2]. The screening row's note names the same train/validation/test shape [13].

**Origin**: built and released by Stanford NLP; both candidate replies are naturally occurring human-written Reddit comments, and the preference label is a collective human signal (comment score and timestamp order), not a model or individual crowdworker judgment [2]. Hub API at the check date: `downloads` 3,321, `downloadsAllTime` 185,862, `likes` 323 [3][6].

**Trained-on-by**: the card names SteamSHP, Stanford's own FLAN-T5 preference models fine-tuned on SHP plus the helpfulness portion of Anthropic's HH-RLHF, reaching 72.8% (SteamSHP-XL, 3B) and 72.0% (SteamSHP-Large, 780M) test accuracy [2]. No other named model or training recipe's adoption of this dataset is stated in the sources fetched for this card.

**Introduced by**: [1] (Ethayarajh, Choi, and Swayamdipta), per the dataset card's own citation request [2].

## Shape

Rows served and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 348,718 |
| `validation` | 18,436 |
| `test` | 18,409 |
| total | 385,563 |

One config, `default`, with 15 columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `post_id` | string |
| `domain` | string |
| `upvote_ratio` | float64 |
| `history` | string |
| `c_root_id_A` | string |
| `c_root_id_B` | string |
| `created_at_utc_A` | int64 |
| `created_at_utc_B` | int64 |
| `score_A` | int64 |
| `score_B` | int64 |
| `human_ref_A` | string |
| `human_ref_B` | string |
| `labels` | int64 |
| `seconds_difference` | float64 |
| `score_ratio` | float64 |

The card's per-subreddit split table sums to the same three split counts across all 18 subreddits (e.g. `askculinary` 45,710/2,094/2,563; `changemyview` 38,173/1,637/1,836; `legaladvice` 21,170/1,106/1,011) [2]. Sizes (datasets-server `/size`) [4]: 827,255,243 bytes of original JSON download, 166,631,371 bytes as Parquet, 752,158,958 bytes decoded in memory. The card states its recommended token limit for finetuning is 512 tokens per input (truncate `history`, not the comments, to fit), and separately reports a maximum length "up to 10.1K T5 tokens" [2]; no source gives a mean or median token count.

## Quality

- The preference label is a collective signal, not an individual judgment: the card states that because comment scores are public on Reddit, a high score is known to influence later voters (citing a herding-effect study), so the label may not match what independent, blind voting would produce [2].
- The card lists explicit inclusion filters used to build each row: the earlier comment must not outscore the later one being compared; the post must be a self-post, pre-2023, unedited, and not NSFW; neither comment nor the post may come from a deleted user, moderator, or the post's own author; the post must score >= 10 and each comment >= 2 [2].
- The card states scraping was capped at 50 comments per post to prevent a few high-comment-count posts from dominating the pairs, and that subreddit-specific abbreviations were expanded (e.g. "CMV" to "Change my view that") while hyperlinks were reduced to their referring text [2].
- The card reports a measured downstream accuracy signal, not a label-quality metric: a FLAN-T5-XL model finetuned on all training data reached 72-73% test accuracy overall (65-80% across individual subreddits), and accuracy rises further when finetuning is restricted to pairs with a larger `score_ratio` [2].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure; none is invented here.

## Load it

Train on `train`, hold out `validation` and `test`, and pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-10-10) [3]. That pin covers what `load_dataset(..., revision=REV)` fetches. The row, split, and byte counts in Shape and the sampled row in A row come from the datasets-server `/size`, `/info`, and `/first-rows` endpoints, which take no revision parameter and answer against the live `main` branch; they are reported here as true as of the check date, not as pinned to `REV` [4][5][12]:

```python
import datasets

REV = "e94b5f32602712d78ed494fe79105b1959396686"  # main at the check date
train = datasets.load_dataset("stanfordnlp/SHP", revision=REV, split="train")           # 348,718 rows
validation = datasets.load_dataset("stanfordnlp/SHP", revision=REV, split="validation")  # 18,436 rows - hold out
test = datasets.load_dataset("stanfordnlp/SHP", revision=REV, split="test")              # 18,409 rows - hold out
```

**Trap**: with no `data_dir` argument, `load_dataset` merges all 18 subreddit directories into the `train`/`validation`/`test` splits above; the card's own example shows loading a single subreddit with `data_dir="askculinary"` instead, which returns a much smaller, single-domain slice of the same three splits [2]. `labels` is randomized so the distribution is roughly 50/50 between A-preferred and B-preferred [2] - do not assume `A` is always the chosen side.

## Neighbors

- `stanfordnlp/SHP-2` - the builder's own successor release: 4.8M pairs (about 3.97M rows served live, with the endpoint marking the count `"partial": true`) across 129 domains, extending SHP with StackExchange posts alongside Reddit; its own card calls it "an extended version of the original 385K SHP dataset" [7][8]. Prefer SHP-2 over this release for larger-scale training; prefer this release when a fixed, fully-counted 385K-row corpus is wanted.
- `jan-hq/shp_dpo_binarized` - this release reformatted into `chosen`/`rejected` role-content message lists for direct DPO use, with `train` deduplicated down to 367,154 rows against this card's 348,718 (a different row count under the same split name) and `test` unchanged at 18,409 rows; its own card gives no processing description beyond the schema [9][10].
- `Dahoas/filtered-SHP` - a filtered subset at 101,430 rows (train 92,238 / test 9,192), far short of this release's 385,563; its `main` branch carries no README, so no source states what filter was applied [11].
- The builder's SteamSHP reward models, `stanfordnlp/SteamSHP-flan-t5-xl` and `stanfordnlp/SteamSHP-flan-t5-large`, were trained on this dataset plus the helpfulness portion of `Anthropic/hh-rlhf` [2]; see that dataset's card for the HH-RLHF side.

## A row

The repository serves one config and one schema across all three splits, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [12], with long text fields truncated:

```json
{
  "post_id": "himc90",
  "domain": "askacademia_train",
  "upvote_ratio": 0.99,
  "history": "In an interview right before receiving the 2013 Nobel prize in physics, Peter Higgs stated that he wouldn't be able to get an academic job today, because he wouldn't be regarded as productive enough. [...]",
  "c_root_id_A": "fwhnqat",
  "c_root_id_B": "fwhp8d4",
  "created_at_utc_A": 1593535113,
  "created_at_utc_B": 1593535824,
  "score_A": 52,
  "score_B": 54,
  "human_ref_A": "Currently wrapping up my PhD. There is a stark difference in work balance life between students in my lab who are focused on industry and those focused on academia. [...]",
  "human_ref_B": "It's ironic to me that research has shown that productivity isn't all it's cracked up to be yet here we are.",
  "labels": 0,
  "seconds_difference": 711.0,
  "score_ratio": 1.0384615385
}
```

`domain` names both the subreddit and the split (`_train`), and `labels` here is 0, meaning `human_ref_B` (the later, higher-scoring comment) is the preferred reply over `human_ref_A` in this row [2][12].

## Where it came from

Built and released by Stanford NLP by scraping Reddit. The card describes the pipeline in three stages [2]:

1. **Domain selection**: 18 subreddits chosen by subscriber count (>= 100K), whether posts pose a question or instruction, whether replies are valued for helpfulness, and whether replies must be grounded in some objectivity rather than pure personal experience.
2. **Post and comment collection**: because Reddit limits access beyond the top 1000 posts per subreddit, the builders started from the top-scoring 1000 all-time posts per subreddit and searched for the 25 most similar posts to each via Reddit's search function, yielding up to 7,500 candidate post IDs per subreddit; scraping was capped at 50 comments per post.
3. **Pair filtering**: a preference A > B was kept only when A was written no later than B and scored higher, the post was a pre-2023, unedited, non-NSFW self-post, neither comment nor the post came from a deleted user/moderator/the post's own author, and the post scored >= 10 with each comment >= 2; train/validation/test splits were built by dividing each subreddit's post IDs 90%/5%/5% so no post spans multiple splits.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision `sha` returned by the Hub API [3]. The datasets-server endpoints ([4], [5], [12]) take no revision parameter and always answer against the live `main` branch, so the row/split/byte counts and the sampled row they supply are current as of the check date only, not covered by that pin.

[1] Ethayarajh, Choi, and Swayamdipta, "Understanding Dataset Difficulty with V-Usable Information", ICML 2022 (Proceedings of Machine Learning Research, vol. 162, pp. 5988-6008). https://proceedings.mlr.press/v162/ethayarajh22a.html - the paper the dataset card asks users to cite; current title read from the live abstract page, which is about V-usable information as a model-relative difficulty measure, not SHP construction specifically. Fetched 2026-08-11.

[2] stanfordnlp/SHP dataset card (README). https://huggingface.co/datasets/stanfordnlp/SHP/raw/main/README.md - summary, data structure, per-subreddit split table, filtering criteria, preprocessing, SteamSHP results, biases/limitations, licence text, contact, citation. Fetched 2026-08-11.

[3] Hugging Face Hub API record for stanfordnlp/SHP. https://huggingface.co/api/datasets/stanfordnlp/SHP?full=true - gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=stanfordnlp%2FSHP Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=stanfordnlp%2FSHP Fetched 2026-08-11.

[6] Hugging Face Hub API record for stanfordnlp/SHP, `downloadsAllTime` expansion. https://huggingface.co/api/datasets/stanfordnlp/SHP?expand%5B%5D=downloadsAllTime Fetched 2026-08-11.

[7] stanfordnlp/SHP-2 dataset card (README). https://huggingface.co/datasets/stanfordnlp/SHP-2/raw/main/README.md - "4.8M collective human preferences", 129 domains, description as an extended version of this dataset. Fetched 2026-08-11.

[8] datasets-server size endpoint for stanfordnlp/SHP-2. https://datasets-server.huggingface.co/size?dataset=stanfordnlp%2FSHP-2 - live row count, marked `"partial": true` by the endpoint. Fetched 2026-08-11.

[9] jan-hq/shp_dpo_binarized dataset card metadata (README front matter; the body has no written description). https://huggingface.co/datasets/jan-hq/shp_dpo_binarized/raw/main/README.md - `chosen`/`rejected` schema, split sizes from `dataset_info`. Fetched 2026-08-11.

[10] datasets-server size endpoint for jan-hq/shp_dpo_binarized. https://datasets-server.huggingface.co/size?dataset=jan-hq%2Fshp_dpo_binarized Fetched 2026-08-11.

[11] datasets-server size endpoint for Dahoas/filtered-SHP. https://datasets-server.huggingface.co/size?dataset=Dahoas%2Ffiltered-SHP - row counts; the repository's `main`-branch README was requested and returned "Entry not found", so no processing description is available. Fetched 2026-08-11.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=stanfordnlp%2FSHP&config=default&split=train Fetched 2026-08-11.

[13] The corpus screening row for `stanfordnlp/SHP`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as preference-pair training data, holding out `validation` and `test`. The card's own scope statement decides the shape - it frames SHP for RLHF reward-model and NLG-evaluation training, not SFT, and flags that the label reflects helpfulness rather than factuality or harmlessness (quoted in the opening paragraph) [2] - and the screening row's note matches this [13].

### The screening row

The row's own note [13]: "385k Reddit preference pairs across 18 subreddits, scored by human votes; train/validation/test." The row carries no flag.
