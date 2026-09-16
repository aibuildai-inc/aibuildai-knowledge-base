# anon8231489123/ShareGPT_Vicuna_unfiltered

94,145 human/model conversation turns worth of cleaned ShareGPT.com dialogue - bare JSON files, not a Hub-viewer-readable dataset - carrying two content-filtering choices for training an unfiltered Vicuna-style model.

**anon8231489123/ShareGPT_Vicuna_unfiltered** repackages user-shared ChatGPT conversations scraped from ShareGPT.com into the training data behind LMSYS's original Vicuna release, which describes fine-tuning LLaMA on "user-shared conversations collected from ShareGPT" [1]. The repository's own card states the pipeline starts from roughly 100k scraped conversations and narrows them to about 53k by removing non-English text, excessive Unicode, excessive repeated characters, and turns containing a long list of "AI Moralizing" refusal phrases, then splits long conversations into 2048-token chunks per FastChat's cleaning recipe [2][3]. **The repository ships two parallel full files with no viewer-readable split/config: one (`..._cleaned_split.json`) keeps turns that contain the phrase "I'm sorry, but", the other (`..._cleaned_split_no_imsorry.json`) has those turns further filtered out; the card leaves the choice to the trainer and warns that the stricter file may remove valuable data** [2]. `datasets-server` cannot parse either file (info/size endpoints return 500, "No (supported) data files found"), so any run must fetch the raw JSON itself [4][5]. It lives at https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered .

**Use it for**: SFT on human-prompt/model-response chat turns (the `conversations` list of `from`/`value` dicts, where `from` is `human` or `gpt`) - the SFT method card's chat format, built by concatenating each conversation's turns; pick one of the two served files per the "I'm sorry, but" trade-off above, and do not merge both into one training set since they are largely the same underlying conversations at different filtering strictness [2]. Not preference data - there is no chosen/rejected pairing anywhere in this schema.

**Licence**: Apache-2.0 (`license: apache-2.0` in the README front matter and the Hub `license:apache-2.0` tag) [2][6]. Ungated, public (`"gated": false`, `"private": false`) [6]. The one catch: this is a third-party re-release of ShareGPT.com user conversations and OpenAI ChatGPT completions, and the card documents no rights clearance from ShareGPT.com, its users, or OpenAI for that content - the apache-2.0 tag covers the repackaging, not a verified chain of title to the underlying conversations [2].

**Shape**: no config/split structure the Hub can read (`datasets-server` `/info` and `/size` both return 500) [4][5]; two top-level JSON-array files at the repo root, each a flat list of `{"id", "conversations"}` objects, counted directly from the served files below.

**Hold out**: not stated. No source for this dataset or for Vicuna's evaluation states a held-out ShareGPT split or documents overlap with a public benchmark; nothing here is checked against any eval set.

**Origin**: built and released by Hub user `anon8231489123`, applying LMSYS FastChat's cleaning scripts to a ShareGPT.com scrape; prompts are human ShareGPT users, responses are OpenAI ChatGPT completions as shared by those users [1][2][3]. Hub API at the check date: `downloads` 412,613, `downloadsAllTime` 2,069,070, `likes` 910 [6][7].

**Trained-on-by**: LMSYS's Vicuna-13B, whose introduction blog post states it was fine-tuned on "user-shared conversations collected from ShareGPT" and gives the earlier, pre-this-repository figure of "around 70K conversations from ShareGPT.com" for that collection effort [1]. No source ties a *specific* commit of this particular Hub repository (as opposed to the underlying ShareGPT scrape it repackages) to Vicuna training; no other named model or recipe citing this exact repository was found.

**Introduced by**: no paper - the dataset card [2] (plus the LMSYS Vicuna blog post that introduces the underlying ShareGPT training data this repository repackages) [1].

## Shape

`datasets-server` cannot parse this repository: `/info` returns `{"error":"No (supported) data files found..."}` (HTTP 500) and `/size` fails the same way, because the served files are bare top-level JSON arrays rather than Parquet-convertible splits [4][5]. Counts below come from downloading the two root-level candidate training files in full and counting directly (read at commit `192ab2185289094fc556ec8ce5ce1e8e587154ca`, the current `main` sha) [6]:

| file | bytes | `"id"` entries | unique base conversation ids | human turns | gpt turns |
| --- | --- | --- | --- | --- | --- |
| `ShareGPT_V3_unfiltered_cleaned_split.json` | 672,837,942 | 94,145 | 50,142 | 333,941 | 367,790 |
| `ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json` | 670,505,514 | 94,145 | 50,142 | 331,410 | 365,184 |

Both files hold the same 50,142 underlying conversations (matched by the base id before its `_<n>` split suffix) at the same 94,145 post-split-chunking entry count; they differ only in how many turns survive the "I'm sorry, but" removal pass, which strips 2,531 human turns and 2,606 gpt turns relative to the plain cleaned file. Every one of the 94,145 entries in both files carries an `_<n>` id suffix (e.g. `QWJhYvA_0`), meaning FastChat's `split_long_conversation.py` chunking step ran over the full 50,142-conversation set, including conversations short enough to become a single `_0` chunk [8][2]. Grepping the literal phrase "I'm sorry, but" gives 2,475 remaining occurrences in the plain cleaned file and 35 residual occurrences in the file named for removing it - the filter is not exhaustive [8].

The repository also ships a `HTML_cleaned_raw_dataset/` subdirectory with four further files - `sg_90k_part1.json` (921,586,083 bytes), `sg_90k_part1_html_cleaned.json` (551,420,693 bytes), `sg_90k_part2.json` (932,290,176 bytes), `sg_90k_part2_html_cleaned.json` (548,183,942 bytes) - an earlier pipeline stage before the language/repeated-character/moralizing filters that produce the two root-level files above; the byte-range sample of `sg_90k_part1.json` and its `_html_cleaned` counterpart both start from the same non-English (Dutch) conversation id `Og9h3C1`, confirming this subdirectory is pre-language-filtering [9][2]. Row counts for these four files were not independently counted; nothing here should be assumed about them beyond their listed byte sizes.

No source states token or sequence-length statistics for any file in this repository; the README's only length-related fact is the 2048-token split-chunk target for `split_long_conversation.py` [2][8], and the script's own default constant reads `max_length=2304` rather than 2048 [8].

## Quality

- The README's own account of the cleaning steps that separate the ~100k raw scrape from the ~53k that entered the two root JSON files: removing non-English conversations, removing "excessive unicode (indicative of Chinese or Korean text, usually)", removing excessive repeated characters, and removing conversations containing any of roughly 130 listed "AI Moralizing" phrases such as "as an AI language model" and "I cannot fulfill your request" [2].
- FastChat's `optional_clean.py`, the script this pipeline step names, implements the non-English filter as: skip if the fraction of non-ASCII characters in the conversation text exceeds 5%, or if language detection (`langdetect`) disagrees with a target language argument; it implements the repeated-character filter as a regex match for any single digit repeated 9 or more times in a row (`re.search(r"(\d)\1{8}", val)`); and its unwanted-word blacklist matches the README's phrase list [8].
- FastChat's `clean_sharegpt.py`, the HTML-to-markdown step, additionally drops any conversation with one turn or fewer, and drops any conversation containing the literal strings "openai" or "chatgpt" in a turn's text [8].
- The two root JSON files still contain the literal phrase "I'm sorry, but" 2,475 times (plain) and 35 times (the file named for its removal), read from the full served files - the removal is not complete [8].
- No source states a measured contamination rate against a benchmark, a duplicate-conversation rate, or an inter-annotator/quality-check figure; none is invented here.

## Load it

`datasets.load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")` fails, because the repo has no Parquet-convertible config for the Hub loader to auto-detect (this is the same failure `datasets-server` reports) [4][5]. Load the JSON file directly instead, pinned to the commit this card's counts were read at:

```python
import datasets

REV = "192ab2185289094fc556ec8ce5ce1e8e587154ca"  # main at the check date
ds = datasets.load_dataset(
    "json",
    data_files=f"https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/{REV}/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
    split="train",
)  # 94,145 rows
```

**Trap**: swap in `ShareGPT_V3_unfiltered_cleaned_split.json` (no `_no_imsorry` suffix) for the version that keeps "I'm sorry, but" refusal-adjacent turns - both files exist in the same commit and neither is a Hub "split" the loader will discover on its own; the choice has to be made in `data_files` [2]. The `HTML_cleaned_raw_dataset/` files are an earlier, pre-language-filtering pipeline stage with raw HTML still embedded in `value` fields, not an alternative training file [9][2].

## Neighbors

- `RyokoAI/ShareGPT52K` - an earlier, upstream release of the same ShareGPT.com scrape by a different uploader ("Ronsor Labs"), cc0-1.0 licensed; its README states the repository "now contains the new 90K conversations version" with the earlier 52K version kept under an `old/` directory, and its first data instance is an English SAS-programming question with a raw-HTML `<div class="markdown...">` GPT response - i.e. still HTML-encoded, upstream of this repository's cleaning [10]. `datasets-server` also cannot size this repository (`/size` returns an empty, failed config) [11].
- `theblackcat102/sharegpt-english` - a further-processed English-only ShareGPT release, license `other`; `datasets-server` reports 50,496 rows in three columns across a single `train` split, a Parquet-convertible repository unlike this one [12].
- This corpus prefers this repository (`anon8231489123/ShareGPT_Vicuna_unfiltered`) over both neighbors when English SFT chat data cleaned specifically toward the Vicuna "unfiltered" recipe (moralizing-phrase removal, no OpenAI/ChatGPT self-mentions) is wanted; reach for `RyokoAI/ShareGPT52K` only if the raw, unfiltered, multilingual scrape is needed instead, and for `theblackcat102/sharegpt-english` only if a Hub-viewer-readable, already-Parquet-converted English subset is required.

## A row

Both root-level training files (`..._cleaned_split.json` and `..._cleaned_split_no_imsorry.json`) share one schema - a top-level array of `{"id", "conversations": [{"from", "value"}, ...]}` objects - so one sample row covers both. From `ShareGPT_V3_unfiltered_cleaned_split.json`, entry index 0 (byte-range fetch of the served file at commit `192ab2185289094fc556ec8ce5ce1e8e587154ca`) [8]:

```json
{
  "id": "QWJhYvA_0",
  "conversations": [
    {
      "from": "human",
      "value": "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients..."
    },
    {
      "from": "gpt",
      "value": "Here are the main ideas of Jeff Walker's Product Launch Formula that can be applied by a growth marketing agency for their clients:\n\n1. Identify the target audience and their needs [...] 8. Use automation: Use technology and automation to streamline the launch process and improve efficiency."
    },
    {
      "from": "human",
      "value": "Summarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients... [...]"
    }
  ]
}
```

The same `id` and first two turns appear identically in `..._cleaned_split_no_imsorry.json`'s entry 0, since this particular conversation contains no "I'm sorry, but" text to strip [8].

## Where it came from

Conversations originate from ShareGPT.com, a now-defunct site where users could publish links to their shared ChatGPT sessions; LMSYS's Vicuna blog post describes collecting "around 70K conversations from ShareGPT.com" for the model this data was built to train [1]. This repository's README describes taking a scrape of roughly 100,000 such conversations and narrowing it to about 53,000 by removing non-English text, excessive Unicode, excessive repeated characters, and turns matching a long "AI Moralizing" refusal-phrase blacklist, then chunking long conversations into token-bounded pieces, following LMSYS FastChat's published data-cleaning recipe [2][3]. The repository bundles the three FastChat scripts that perform this pipeline - `clean_sharegpt.py` (HTML-to-markdown conversion, skips short conversations and ones mentioning "openai"/"chatgpt"), `optional_clean.py` (non-English/Unicode filter, repeated-character filter, moralizing-phrase blacklist), and `split_long_conversation.py` (token-bounded chunk splitting, appending `_<start_idx>` to each chunk's id) - and its `HTML_cleaned_raw_dataset/` subdirectory holds an intermediate, pre-language-filtering stage of the same scrape [8][9].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] LMSYS, "Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality", 2023-03-30. https://lmsys.org/blog/2023-03-30-vicuna/ - introduces training LLaMA on ShareGPT-collected conversations and gives the "around 70K conversations" figure for that separate collection effort. Fetched 2026-08-11.

[2] anon8231489123/ShareGPT_Vicuna_unfiltered dataset card (README). https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/raw/main/README.md - cleaning-step description, moralizing-phrase list, the two-file choice and its trade-off, licence front matter. Fetched 2026-08-11.

[3] lm-sys/FastChat, "Data cleaning" docs. https://raw.githubusercontent.com/lm-sys/FastChat/main/docs/commands/data_cleaning.md - the cleaning recipe this repository's README names as its procedure. Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=anon8231489123%2FShareGPT_Vicuna_unfiltered - returns HTTP 500, "No (supported) data files found". Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=anon8231489123%2FShareGPT_Vicuna_unfiltered - returns HTTP 500, same cause. Fetched 2026-08-11.

[6] Hugging Face Hub API record for anon8231489123/ShareGPT_Vicuna_unfiltered. https://huggingface.co/api/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered?full=true - licence, gate/private status, `sha`, `downloads`, `likes`, file tree with byte sizes, last-modified date. Fetched 2026-08-11.

[7] Hugging Face Hub API, `downloadsAllTime` expansion. https://huggingface.co/api/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] The dataset's own served files, read directly since datasets-server cannot parse them: full downloads of `ShareGPT_V3_unfiltered_cleaned_split.json` and `ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json` from https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/ for entry counts, turn counts, and the sample row; and the repository's bundled `clean_sharegpt.py`, `optional_clean.py`, `split_long_conversation.py` scripts (same commit) for the cleaning-pipeline implementation details. Fetched 2026-08-11.

[9] Byte-range samples of `HTML_cleaned_raw_dataset/sg_90k_part1.json` and `sg_90k_part1_html_cleaned.json` at the same commit, showing the pre-language-filtering stage still contains non-English (Dutch) conversations and, in the non-html-cleaned file, raw HTML markup. Fetched 2026-08-11.

[10] RyokoAI/ShareGPT52K dataset card (README) and Hub API record. https://huggingface.co/datasets/RyokoAI/ShareGPT52K/raw/main/README.md and https://huggingface.co/api/datasets/RyokoAI/ShareGPT52K - licence, the 52K-to-90K note, sample row with embedded HTML. Fetched 2026-08-11.

[11] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=RyokoAI%2FShareGPT52K - returns an empty, failed config, meaning this repository is likewise not Hub-viewer-readable. Fetched 2026-08-11.

[12] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=theblackcat102%2Fsharegpt-english - 50,496 rows, 3 columns, one `train` split, Parquet-convertible. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT chat data, with the caveat this card already states: choose one of the two parallel root-level files (with or without "I'm sorry, but" turns) rather than merging both, since they cover the same 50,142 underlying conversations [2][8]. The screening row's own note describes exactly this shape and exactly this trade-off.

### The screening row

The row's own note: "The ShareGPT conversation dump used to train Vicuna, about 100k conversations narrowed to 53k by removing non-English text, excessive unicode, repeated characters and refusal phrases, with a second copy that also strips \"I'm sorry, but\" responses; apache-2.0, the prompts are human and the answers are model-written, and the viewer serves no rows because the repo ships bare json files it cannot read, so a run must fetch the json directly." The row carries no flag.
