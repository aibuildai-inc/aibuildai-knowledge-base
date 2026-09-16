# Amod/mental_health_counseling_conversations

3,512 single-turn question–answer pairs of real mental-health counseling exchanges, each a user question (`Context`) matched to a licensed counselor's written reply (`Response`), served as one JSON file with no predefined splits.

**Amod/mental_health_counseling_conversations** is built and maintained by the Hugging Face user Amod, who describes it as a compilation of real one-on-one mental-health counseling conversations between individuals and licensed professionals, collected from two public counseling websites and cleaned into question–answer pairs suitable for fine-tuning or instruction-tuning dialogue models [1]. No academic paper introduces it; the dataset card is the only source of its provenance [1]. It lives at https://huggingface.co/datasets/Amod/mental_health_counseling_conversations . **The dataset carries a custom RAIL-D licence: non-commercial research use is free, but any commercial use requires a donation of at least USD 100 to a named mental-health charity with proof emailed within 30 days, and individual question–answer pairs must not be rewritten, deleted, or altered (filtering/subsetting rows is permitted) [1].**

**Use it for**: single-turn SFT chat data - each row maps directly to a one-turn instruction/response pair (`Context` as the user turn, `Response` as the assistant turn); this is the SFT method card's shape, not a preference-pair format. Respect the no-rewrite clause above: row-level filtering is allowed, but editing the text of a kept row is not [1].

**Licence**: custom agreement ("RAIL-D", Hub `license: other`), ungated [1][2]. The catch: commercial use is conditioned on a mandatory USD 100+ donation with proof of payment, and content may be filtered but not rewritten [1].

**Shape**: 3,512 rows, one config (`default`), one split (`train`), two string columns (`Context`, `Response`) [3][4].

**Hold out**: nothing declared. The dataset ships no predefined splits and no eval config, and the corpus screening row records no contamination flag [5][6].

**Origin**: built by Amod from real counselor replies scraped from two unnamed public counseling websites, per the dataset's own card; the questions are real user submissions and the responses are real licensed-professional replies, not model-generated [1]. Hub API at the check date: 2,396 downloads in the current period, 123,933 all-time downloads, 489 likes [2][7].

**Trained-on-by**: 109 Hub models declare this dataset in their tags as of the check date, including the author's own `Amod/falcon7b-fine-tuned-therapy-merged` and third-party fine-tunes such as `GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2`, `Jaykumaran17/Zephyr7b-Beta-sharded-bf16-finetuned-mental-health-conversational-Amod`, and `Felladrin/Llama-68M-Chat-v1` [8].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and splits (datasets-server `/size`) [3]:

| split | rows |
| --- | --- |
| `train` | 3,512 |

One config, `default`, two columns (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `Context` | string |
| `Response` | string |

Sizes (datasets-server `/size`) [3]: 4,790,520 bytes of original JSON download, 2,451,125 bytes as Parquet, 4,643,156 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The dataset card states all personally identifiable information was removed during curation and that the question–answer pairs were "retained verbatim to preserve conversational integrity" (i.e., not rewritten) [1].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset.
- The card's only stated collection detail is that data came "directly from two publicly accessible counselling websites; no private or paid sources were used" [1] - it does not name the two sites.
- The card notes commercial-use terms changed over the dataset's history: it moved from an OpenRAIL licence to the current RAIL-D terms in a 2025-07-06 commit, and the commit's own message describes the change as updating the licensing terms to RAIL-D, allowing commercial use under the donation condition [9].

## Load it

One split, no held-out portion; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date, matching the shortlist's pinned commit) [2]:

```python
import datasets

REV = "d7e86f0813c5690181b41f97403c3674aa55dcef"  # main at the check date
ds = datasets.load_dataset("Amod/mental_health_counseling_conversations", revision=REV, split="train")  # 3,512 rows
```

**Trap**: the repository only ships one raw file, `combined_dataset.json`, with no train/test split defined by the builder - `split="train"` is the datasets-library default name for a single-file JSON dataset, not a curated training subset distinct from a held-out one. There is no split to hold out; if a held-out set is needed for evaluation, it must be carved out manually before training.

## Neighbors

Several third-party re-releases exist; each row count below was read live at the check date [10]:

- `MaggiePai/mental_health_counseling_conversations` - an exact mirror: same 3,512 rows, same two columns, same byte sizes as this repository. Its own README states it is "cloned from" this dataset, and additionally names a source this card's own README does not: Nicolas Bertagnolli's blog post "Counsel chat: Bootstrapping high-quality therapy data" and its accompanying `counsel-chat` GitHub data, which the mirror's Source Data section links as where its two-platform data "was sourced from" [11]. This claim is the mirror's own attribution, not a statement in Amod's card, and is not independently confirmed here.
- `Sulav/mental_health_counseling_conversations_sharegpt` - the same 3,512 rows reshaped into a third `conversations` column in ShareGPT chat-turn format alongside the original `Context`/`Response` columns, for direct use with chat-format SFT trainers [10][12].
- A Hub dataset search for "mental_health_counseling" returns roughly 30 more repositories (`herisan`, `TVRRaviteja`, `chansung`, `tcabanski`, and others) whose names suggest they reformat or subset this same source [10]. Their cards were not opened here, so no claim is made about their content; this original repository is preferred by default for the complete, unmodified pair set.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6]:

```json
{
  "Context": "I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here.\n   I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it.\n   How can I change my feeling of being worthless to everyone?",
  "Response": "If everyone thinks you're worthless, then maybe you need to find new people to hang out with.Seriously, the social context in which a person lives is a big influence in self-esteem. [...] Your feeling of worthlessness may be good in the sense of motivating you to find out that you are much better than your feelings today."
}
```

`Context` holds the user's question as submitted (occasionally spanning several sentences), and `Response` holds the counselor's full written reply.

## Where it came from

Built and released by the Hugging Face user Amod. Per the dataset's own card, the raw question–answer pairs were "collected directly from two publicly accessible counselling websites" that the card does not name, with "no private or paid sources" involved [1]. The card states personally identifiable information was removed during anonymization and that the pairs were kept unedited to preserve their original wording [1]. The repository's commit history shows the licence terms were changed once, from OpenRAIL to the current RAIL-D terms, in a 2025-07-06 commit titled "licensing (#8)"; the commit's own message describes the change as updating the licensing terms to RAIL-D and adding a commercial-use path via donation [9].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Amod/mental_health_counseling_conversations dataset card (README). https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/raw/main/README.md - dataset summary, source-data statement, licence terms, curation rationale. Fetched 2026-08-11.

[2] Hugging Face Hub API record for Amod/mental_health_counseling_conversations. https://huggingface.co/api/datasets/Amod/mental_health_counseling_conversations?full=true - licence field, gate status, `sha`, `downloads`, `likes`, siblings. Fetched 2026-08-11.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Amod%2Fmental_health_counseling_conversations Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Amod%2Fmental_health_counseling_conversations Fetched 2026-08-11.

[5] The corpus screening row for `Amod/mental_health_counseling_conversations`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[6] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Amod%2Fmental_health_counseling_conversations&config=default&split=train Fetched 2026-08-11.

[7] Hugging Face Hub API record with all-time downloads expanded. https://huggingface.co/api/datasets/Amod/mental_health_counseling_conversations?expand[]=downloadsAllTime - `downloadsAllTime` figure. Fetched 2026-08-11.

[8] Hugging Face Hub models API, filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:Amod/mental_health_counseling_conversations&limit=1000 - count and identities of models tagging this dataset. Fetched 2026-08-11.

[9] Hugging Face Hub commits API for this repository. https://huggingface.co/api/datasets/Amod/mental_health_counseling_conversations/commits/main - commit history showing the 2025-07-06 "licensing (#8)" change from OpenRAIL to RAIL-D. Fetched 2026-08-11.

[10] Hugging Face Hub datasets search and size endpoint, for the neighbor row counts and column layouts above: `MaggiePai/mental_health_counseling_conversations`, `Sulav/mental_health_counseling_conversations_sharegpt`, and other search results for "mental_health_counseling". https://huggingface.co/api/datasets?search=mental_health_counseling&limit=30 and https://datasets-server.huggingface.co/size?dataset=<id> - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] MaggiePai/mental_health_counseling_conversations dataset card (README). https://huggingface.co/datasets/MaggiePai/mental_health_counseling_conversations/raw/main/README.md - states it is cloned from this dataset and names Bertagnolli's Counsel Chat blog post and GitHub data as the upstream source. Fetched 2026-08-11.

[12] Sulav/mental_health_counseling_conversations_sharegpt dataset card and first-rows. https://huggingface.co/datasets/Sulav/mental_health_counseling_conversations_sharegpt/raw/main/README.md and https://datasets-server.huggingface.co/first-rows?dataset=Sulav%2Fmental_health_counseling_conversations_sharegpt&config=default&split=train - column layout including the added `conversations` field. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn SFT chat data, with the licence's no-rewrite and commercial-donation terms carried forward as usage restrictions. Two facts decide it, both established above: the card's own licence terms bar commercial use without a paid donation and bar rewriting kept rows [1], and the screening row's note flags the same licence restriction [5].

### The screening row

The row's own note [5]: "real one-on-one counseling question/answer exchanges from licensed professionals; RAIL-D licence bars commercial use and modification." The row carries no flag.
