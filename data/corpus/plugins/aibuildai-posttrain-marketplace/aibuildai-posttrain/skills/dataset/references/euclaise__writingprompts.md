# euclaise/writingprompts

303,358 prompt/story pairs from Reddit's r/WritingPrompts forum, the original WritingPrompts long-form story-generation dataset, re-hosted as Parquet.

**euclaise/writingprompts** is a Hugging Face re-parsing of the WritingPrompts dataset introduced by "Hierarchical Neural Story Generation" [1], which scraped three years of prompt/story pairs from r/WritingPrompts via the official Reddit API and cleaned them by removing bot posts, deleted posts, moderator comments, and stories under 30 words [1]. The dataset card states it was parsed from the original fairseq archive used by that paper [2]; it lives at https://huggingface.co/datasets/euclaise/writingprompts . Each row pairs a `prompt` with a full `story`; the paper's own 5%/5% validation/test reservation is preserved as this release's `validation` and `test` splits [1]. **The paper's train/validation/test split is exactly this release's `train`/`validation`/`test` split (272,600 / 15,620 / 15,138), and the `test` split is the split used across the story-generation literature to evaluate against this benchmark, so it must be held out of training to keep any evaluation on it meaningful.**

**Use it for**: long-form conditional text generation (prompt-to-story), i.e. SFT-style sequence-to-sequence training - feed `prompt` as input and `story` as the target completion. Held-out `test` only for evaluation, never for training. Maps to the SFT method card's plain prompt/completion format, with no chat template applied (`chat_dialect` is none on the shortlist row).

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`), ungated (`"gated": false`, `"private": false`) [3]. No further catch stated beyond the licence grant itself.

**Shape**: 303,358 rows in one config (`default`), splits `train` 272,600 / `test` 15,138 / `validation` 15,620, two string columns (`prompt`, `story`) [4][5].

**Hold out**: `test` (15,138 rows) - it is the paper's original held-out test split and is used elsewhere as a story-generation benchmark, so training on it would contaminate any later evaluation against that benchmark [1].

**Origin**: built by Facebook AI Research and released as the WritingPrompts dataset [1]; this Hub repository is `euclaise`'s re-parsing of that same archive into Parquet [2]. Both prompts and stories are the original human-written Reddit posts and comments, not model-generated [1]. Hub API at the check date: `downloads` 4,859, `downloadsAllTime` 61,854, `likes` 70 [3].

**Trained-on-by**: Hugging Face's model-hub filter for datasets tagging `euclaise/writingprompts` lists several fine-tunes, including `AKeigo/Llama3.1_StoryGeneration` (a Llama-3.1-8B fine-tune for story generation) and `arbazsiddiqui/Ozan-v1-12B` (a Mistral-Nemo-12B creative-writing fine-tune that also mixes in `Gryphe/Opus-WritingPrompts` and other creative-writing sets) [6]. No independent research paper reporting training on this specific Hub release was found.

**Introduced by**: [1] (Fan, Lewis, Dauphin, "Hierarchical Neural Story Generation"); no separate dataset-card blog post - the Hub README [2] is the only re-hosting documentation.

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 272,600 |
| `test` | 15,138 |
| `validation` | 15,620 |
| total | 303,358 |

One config, `default`, two columns (datasets-server `/info`) [4][5]:

| column | dtype |
| --- | --- |
| `prompt` | string |
| `story` | string |

Sizes (datasets-server `/size`) [5]: 605,049,830 bytes of Parquet download, 958,071,071 bytes decoded in memory. The origin paper states word-level statistics for the whole 303,358-row collection: 7.7M total prompt words and 200M total story words, averaging 28.4 words per prompt and 734.5 words per story [1]. It does not give a token count under any specific tokenizer.

## Quality

- The origin paper describes its own cleaning pass: bot posts, deleted posts, special announcements, and moderator comments were removed, and stories under 30 words were dropped; the paper also states stories must avoid general profanity and inappropriate content and should be inspired by the prompt, though a story does not necessarily have to fulfill every requirement of the prompt [1].
- The paper tokenized both prompts and stories with NLTK before release [1]; the served rows still carry that tokenization as visible whitespace around punctuation and contractions (e.g. `[ WP ]`, `` `` ``, `'s`), confirmed by reading the first five rows of each split via datasets-server `first-rows` [7][8][9].
- No source states a measured contamination rate, deduplication rate, or annotator-agreement figure for this release; none is invented here. The Hub README carries no quality discussion beyond naming its source archive [2].

## Load it

Train on `train`, evaluate on `test` only, never train on it, and pin the revision this card's numbers were read at:

```python
import datasets

REV = "35f0aa359452ba8147b34d925684fccee26679cc"  # main at the check date
train = datasets.load_dataset("euclaise/writingprompts", revision=REV, split="train")           # 272,600 rows
validation = datasets.load_dataset("euclaise/writingprompts", revision=REV, split="validation")  # 15,620 rows
test = datasets.load_dataset("euclaise/writingprompts", revision=REV, split="test")              # 15,138 rows - hold out from training
```

**Trap**: the text is NLTK-pretokenized with spaces around punctuation and contractions rather than natural running text (see A row, below); training or evaluating without detokenizing first will teach or score a model on this tokenized surface form rather than normal prose [1][7]. The `revision` pin above reproduces the actual Parquet files `load_dataset` reads, but the row counts, byte sizes, and schema reported in Shape and Origin come from the datasets-server `/info`, `/size`, and `/first-rows` endpoints, which take no revision parameter and always answer for the current `main` [4][5][7][8][9]; they matched the pinned commit's own `dataset_info` metadata [2] at the check date, but are not themselves pinned.

## Neighbors

Same builder (`euclaise`), different shapes - not simple duplicates of this release, each read live at the check date [10]:

- `euclaise/WritingPromptsX` - 1,245,546 rows of raw r/WritingPrompts *comments* (columns `post_title`, `body`, `score`, `gilded`, `post_score`), scraped via PushShift through December 2022; its own card describes it as "inspired by" this dataset but "a bit more complete" [10][11].
- `euclaise/WritingPrompts_preferences` - 265,174 rows of human preference data: one post plus a list of comment texts, scores, and times per row, not a chosen/rejected pair [10][12].
- `euclaise/WritingPrompts_binarized` - 352,994 rows, the preferences set reprocessed into explicit `chosen`/`rejected` pairs "like SHP" per its own card, for preference-pair training rather than SFT [10][12].
- `euclaise/WritingPrompts_curated` - 66,332 rows of `prompt`/`body` pairs with `post_score` and `comment_score` kept, a smaller curated subset of the comments data [10][12].
- `Gryphe/Opus-WritingPrompts` (different builder) - 6,022 rows of synthetic short stories generated by Claude Opus using the same subreddit's prompts as seeds, 4,000-6,000 characters each per its own card, plus a GPT-3.5 negative-counterpart variant for KTO training; not a re-hosting of this dataset but a synthetic reworking of the same source community [13][14].

This corpus prefers this original human-written release for plain prompt-to-story SFT; reach for `WritingPrompts_preferences`/`_binarized` for preference-pair training on the same community's data, and `WritingPromptsX`/`_curated` only if raw unpaired comments are needed.

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], with the story truncated:

```json
{
  "prompt": "[ WP ] You 've finally managed to discover the secret to immortality . Suddenly , Death appears before you , hands you a business card , and says , `` When you realize living forever sucks , call this number , I 've got a job offer for you . ''",
  "story": "So many times have I walked on ruins, the remainings of places that I loved and got used to.. At first I was scared, each time I could feel my city, my current generation collapse, break into the black hole that thrives within it, I could feel humanity, the way I'm able to feel my body.. After a few [...]"
}
```

Both splits share this same `prompt`/`story` schema; reading the first five rows of `test` and `validation` confirms the same two-column shape and the same NLTK-tokenized punctuation spacing in `prompt` [8][9].

## Where it came from

Built and released by Facebook AI Research, scraped from Reddit's r/WritingPrompts community over three years via the official Reddit API, where users post story premises ("prompts") and other users respond with full stories [1]. The collection was cleaned by removing automated bot posts, deleted posts, special announcements, and moderator comments, and by dropping stories under 30 words; both prompts and stories were tokenized with NLTK before release [1]. This Hub repository is a re-parsing of that same released archive (`dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz`) into a Parquet-backed Hub dataset, per its own README [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Fan, Lewis, Dauphin, "Hierarchical Neural Story Generation", 2018. https://arxiv.org/abs/1805.04833 - the origin paper: collection method, cleaning rules, split sizes and word-count statistics, current title read from the live abs page. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/1805.04833). Fetched 2026-08-11.

[2] euclaise/writingprompts dataset card (README). https://huggingface.co/datasets/euclaise/writingprompts/raw/main/README.md - source-archive attribution, load instructions, licence and split metadata. Fetched 2026-08-11.

[3] Hugging Face Hub API record for euclaise/writingprompts. https://huggingface.co/api/datasets/euclaise/writingprompts?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=euclaise%2Fwritingprompts - this endpoint takes no functioning revision parameter (verified: passing `&revision=` an arbitrary nonexistent value returns the same payload), so the schema it reports is a live read, not one pinned to the `revision` used in Load it. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=euclaise%2Fwritingprompts - same live, unpinned behavior as [4]: the row counts and byte sizes above are current as of the fetch date, not guaranteed to match the pinned `revision` in Load it. Fetched 2026-08-11.

[6] Hugging Face model-hub API filtered by `dataset:euclaise/writingprompts`. https://huggingface.co/api/models?filter=dataset:euclaise/writingprompts&limit=30 - this endpoint takes no revision parameter, so it is a live, unpinned list of adoption evidence. Fetched 2026-08-11.

[7] datasets-server first-rows endpoint, `train`. https://datasets-server.huggingface.co/first-rows?dataset=euclaise%2Fwritingprompts&config=default&split=train - live and unpinned, same as [4][5]. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, `test`. https://datasets-server.huggingface.co/first-rows?dataset=euclaise%2Fwritingprompts&config=default&split=test - live and unpinned, same as [4][5]. Fetched 2026-08-11.

[9] datasets-server first-rows endpoint, `validation`. https://datasets-server.huggingface.co/first-rows?dataset=euclaise%2Fwritingprompts&config=default&split=validation - live and unpinned, same as [4][5]. Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor, for every neighbor row count above: `euclaise/WritingPromptsX`, `euclaise/WritingPrompts_preferences`, `euclaise/WritingPrompts_binarized`, `euclaise/WritingPrompts_curated`, `Gryphe/Opus-WritingPrompts`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] euclaise/WritingPromptsX dataset card (README). https://huggingface.co/datasets/euclaise/WritingPromptsX/raw/main/README.md Fetched 2026-08-11.

[12] Neighbor dataset cards (READMEs), read for what each same-builder re-release says it did: `euclaise/WritingPrompts_preferences`, `euclaise/WritingPrompts_binarized`, `euclaise/WritingPrompts_curated`. https://huggingface.co/datasets/<id>/raw/main/README.md Fetched 2026-08-11.

[13] Gryphe/Opus-WritingPrompts dataset card (README). https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts/raw/main/README.md Fetched 2026-08-11.

[14] datasets-server size endpoint for Gryphe/Opus-WritingPrompts. https://datasets-server.huggingface.co/size?dataset=Gryphe%2FOpus-WritingPrompts Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT-style prompt-to-story training, with the `test` split held out. Two facts already established above decide it: the release's `train`/`validation`/`test` split is exactly the origin paper's own 5%/5% held-out reservation [1], and the screening row's note identifies the `test` split as the story-generation benchmark that later work evaluates against, so it must stay out of training [15].

### The screening row

The row's own note [15]: "Reddit r/WritingPrompts prompt-to-story pairs; train/validation/test, and the test split is the standard story-generation benchmark." The row carries no flag.

[15] The corpus screening row for `euclaise/writingprompts`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.
