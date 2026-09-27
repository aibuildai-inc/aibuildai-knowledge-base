# mb7419/legal-advice-reddit_preference

70,324 preference pairs built from r/legaladvice: each post paired with a higher-scored and a lower-scored reply, over 24,986 distinct posts - human-written, score-ranked, unlicensed, and full of personal situations.

**mb7419/legal-advice-reddit_preference** has a README holding only its `dataset_info` block [1]. Its columns - `post_id`, `post_score`, `post`, `chosen_ans`, `rejected_ans`, `chosen_score`, `rejected_score` - show the construction: for a Reddit post, two replies are paired and the higher-scored one is `chosen` [2]. In all 77 rows the viewer served, `chosen_score` exceeds `rejected_score` [3]. The `post_id` values are Reddit ids (row 0: `1al05o`) [3]. It lives at https://huggingface.co/datasets/mb7419/legal-advice-reddit_preference .

**Use it for**: preference data for lay legal advice, ranked by community votes. Votes reward helpful, well-received replies, not legally correct ones, so the pairs teach tone and engagement as much as law.

**Licence**: none on the card [4][1]. Reddit content is subject to Reddit's own terms; no source here states a grant. The one catch: no licence, and the posts describe real people's legal and medical situations.

**Shape**: 70,324 rows in one `train` split [5]; seven columns [2]. Measured: 24,986 distinct `post_id` values, so each post contributes on average about 2.8 pairs [6].

**Hold out**: no split is set aside. Split by `post_id`, not by row, or the same post lands on both sides. LegalBench's `learned_hands_*` tasks also come from r/legaladvice posts [7], so screen before reporting them.

**Origin**: posts and replies written by Reddit users; the pairing rule is inferred from the columns, not documented [2][3]. Hub API at the check date: `downloads` 14, `downloadsAllTime` 454, `likes` 0 [4].

**Trained-on-by**: no model on the Hub declares this dataset through its dataset tag [8].

**Introduced by**: no paper and no card text [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 70,324 |
| total | 70,324 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `post_id` | string |
| `post_score` | float64 |
| `post` | string |
| `chosen_ans` | string |
| `rejected_ans` | string |
| `chosen_score` | float64 |
| `rejected_score` | float64 |

## Quality

- Measured duplication on the full split: 55 rows (0.08%) repeat another row exactly [6].
- Text has had punctuation stripped or spaced out in places - row 0's post contains "r askreddit" and "http:  www.reddit.com r AskReddit" [3] - so URLs and some punctuation are damaged.
- Row 0 is a personal, sensitive family and medical situation [3]; the dataset carries no anonymisation statement.

## Load it

Split by post, pin the revision:

```python
import datasets

REV = "7e274e709c468cf302683dd7a986c54a23a64252"  # main at the check date
ds = datasets.load_dataset("mb7419/legal-advice-reddit_preference", revision=REV, split="train")  # 70,324 rows
ds = ds.rename_columns({"post": "prompt", "chosen_ans": "chosen", "rejected_ans": "rejected"})
posts = sorted(set(ds["post_id"]))
held = set(posts[: len(posts) // 20])
train = ds.filter(lambda r: r["post_id"] not in held)
```

**Trap**: a random row split leaks: 64.47% of rows share their `post_id` with another row [6], so a row-level test set is mostly posts the model trained on. Split on `post_id`.

## A row

From `config="default"`, `split="train"`, row 1 (datasets-server `/first-rows`) [3] - row 0 is a sensitive personal disclosure and is not reproduced here - truncated:

```json
{
  "post_id": "1chdxr",
  "post_score": 44.0,
  "post": "Is this even legal...? After some Twitter stuff happened at my school  I am junior by the way  I was called into the principals office and told to hand over my phone which I did. Then he told me I have to give him my pass code  which I politely denie [...]",
  "chosen_ans": "You likely don t need to give anyone your password.  If the cops show up  demand the phone back and specifically state that you don t consent to searches  seizures or questioning without an attorney.  Note that the change to a private Twitter account [...]",
  "rejected_ans": "What state are you in?  Public or private school?",
  "chosen_score": 57.0,
  "rejected_score": 3.0
}
```

## Where it came from

Uploaded by user mb7419 in February 2024 [4]; the pairing rule and the source dump of r/legaladvice are not documented [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] mb7419/legal-advice-reddit_preference dataset card (README). https://huggingface.co/datasets/mb7419/legal-advice-reddit_preference/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mb7419%2Flegal-advice-reddit_preference - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=mb7419%2Flegal-advice-reddit_preference&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] Hugging Face Hub API record for mb7419/legal-advice-reddit_preference. https://huggingface.co/api/datasets/mb7419/legal-advice-reddit_preference?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mb7419%2Flegal-advice-reddit_preference - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] This skill's own duplication count on the full `train` split at the pinned revision - whole-row and `post_id` uniqueness. Run 2026-09-23.

[7] nguha/legalbench dataset card (README). https://huggingface.co/datasets/nguha/legalbench/raw/main/README.md - "Several tasks have been derived from the LearnedHands corpus, which consists of public posts on /r/LegalAdvice". Fetched 2026-09-23.

[8] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:mb7419/legal-advice-reddit_preference&sort=downloads - live list, unpinned. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable only with care: vote-ranked human preferences for lay legal advice, with no licence, sensitive personal content, and damaged punctuation. Split by `post_id`.

### The screening row

The row's own note: "r/legaladvice replies paired by score; no licence; sensitive posts." The row carries the flag `no-licence`.
