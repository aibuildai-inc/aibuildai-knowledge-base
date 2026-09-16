# UCSD26/medical_dialog

A patient-doctor consultation corpus in English and Chinese, scraped from three online medical-consultation sites, served as four differently-shaped Hugging Face configs totalling 5,558,895 declared rows.

**UCSD26/medical_dialog** is the Hugging Face packaging of MedDialog, introduced in "MedDialog: Two Large-scale Medical Dialogue Datasets" [1]: an English subset of conversations from healthcaremagic.com and icliniq.com, and a Chinese subset from haodf.com, each consultation carrying a patient description and a doctor-patient turn sequence [2]. The repository ships as a `datasets` loading script rather than served Parquet, and two of its four configs (`en`, `zh`) additionally require a manual Google Drive download before they will load at all [2][3]. **The source sites' own terms of service, quoted on the dataset card, prohibit scraping, republishing, duplicating, or reselling the underlying content "for commercial or any other purpose whatsoever," and the Hub license tag is `unknown`, not a permissive grant** [2][4]. It lives at https://huggingface.co/datasets/UCSD26/medical_dialog .

**Use it for**: multi-turn dialogue SFT after reshaping - the raw `en`/`zh` configs give a `dialogue_turns` sequence of `{speaker, utterance}` that maps to the SFT chat method card's role/content list once `Patient`/`病人` and `Doctor`/`医生` are relabelled to `user`/`assistant`; the `processed.*` configs give a flat `description` + `utterances` list-of-strings that needs the same turn-role reconstruction. Restated restriction: the source sites bar commercial scraping/republication and the license is unmarked, so treat this as research-use data pending your own license review [2][4].

**Licence**: tag is `unknown` (not SPDX-mapped), repository is ungated and public [4]. The one catch: the card quotes both icliniq.com's and healthcaremagic.com's terms of service explicitly forbidding scraping, republishing, duplicating, or reselling their content "for commercial or any other purpose whatsoever" [2].

**Shape**: 4 configs - `en` (229,674 rows, one `train` split), `zh` (1,921,127 rows, one `train` split), `processed.en` (603 rows: 482/60/61 train/validation/test), `processed.zh` (3,407,491 rows: 2,725,989/340,748/340,754 train/validation/test) - summing to the card's declared 5,558,895 [4].

**Hold out**: `processed.en` `test` (61 rows) and `processed.zh` `test` (340,754 rows) are the card's own held-out splits for those two configs [2][4]. The raw `en` and `zh` configs carry only a single `train` split with no card-declared test/validation portion at all - the card states there "are no data splits on the original raw data" [2] - so a reader scoring against `en` or `zh` must carve their own holdout before a scored run.

**Origin**: built and released by UCSD (source_datasets: original) [4]; content is human-authored (real patient and doctor posts found on the three consultation sites), not model-generated [2]. Hub API as of the check date: `downloads` 1,054, `downloadsAllTime` 41,576, `likes` 176 [4].

**Trained-on-by**: none found. Of the titles of the 50 most-recent papers citing the origin paper [1] fetched for this card via a citation-graph lookup (a first page of an unknown, larger total) [5], none names a specific downstream model or training recipe as having trained on this exact Hub release.

**Introduced by**: [1] (Chen et al., "MedDialog: Two Large-scale Medical Dialogue Datasets").

## Shape

Four configs, from the card's own `dataset_info` block [4]:

| config | split | rows | columns |
| --- | --- | --- | --- |
| `en` | train | 229,674 | `file_name`, `dialogue_id`, `dialogue_url`, `dialogue_turns` (sequence of `speaker`{Patient,Doctor}, `utterance`) |
| `zh` | train | 1,921,127 | same fields, `speaker`{病人,医生} |
| `processed.en` | train / validation / test | 482 / 60 / 61 | `description` (string), `utterances` (sequence of string) |
| `processed.zh` | train / validation / test | 2,725,989 / 340,748 / 340,754 | `utterances` (sequence of string) only, no `description` |

Total: 5,558,895 rows, matching the card's declared count exactly [4]. In-memory `dataset_size`: `en` 290,274,759 bytes, `zh` 1,092,063,621 bytes, `processed.en` 469,404 bytes, `processed.zh` 1,964,906,402 bytes; the card records `download_size: 0` for the raw `en`/`zh` configs because their files are not fetched by the script itself but by a manual Google Drive download, and `download_size` 524,214 bytes / 2,082,354,155 bytes for `processed.en` / `processed.zh`, whose files the script downloads automatically [4]. No source states sequence-length or token statistics for any config; none is invented here.

## Quality

- The card's own "Personal and Sensitive Information," "Annotation process," "Who are the annotators?," and "Initial Data Collection and Normalization" sections are all left as "[More Information Needed]" - no source documents a PII-redaction process, an annotation protocol, or scraping methodology beyond naming the three source sites [2].
- No source states a measured contamination rate, duplicate rate, or label-agreement figure for this dataset; none is invented here.
- The `en`/`zh` loading-script parser is line-heuristic (it matches literal markers such as `"id"`, `"Description"`, `"Dialogue"`, and Chinese colon patterns to segment each consultation), and the script's own comments flag this as fragile, noting turns are truncated to an even count and multi-line English utterances are reassembled by continuation heuristics [3].
- The Hub API's `cardData` sets `"viewer": false` for this repository, and all four datasets-server endpoints (`/info`, `/size`, `/splits`) return HTTP 501 "dataset viewer is disabled" [4][6]; the reason is that this is a code-executing loading script, the same condition Hugging Face reports for the sibling `bigbio/meddialog` repository ("this dataset runs arbitrary python code") [7].

## Load it

The repository has no Parquet export - `load_dataset` runs `medical_dialog.py` and needs `trust_remote_code=True` [2][3]. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-09-18) [4]:

```python
import datasets

REV = "1c598b66d82a3ba91e1cd1214e91d0888b1474d8"  # main at the check date

# processed configs download automatically from Google Drive - these work unmodified
proc_en = datasets.load_dataset("UCSD26/medical_dialog", name="processed.en", revision=REV, trust_remote_code=True)
proc_zh = datasets.load_dataset("UCSD26/medical_dialog", name="processed.zh", revision=REV, trust_remote_code=True)

# raw configs require a MANUAL prior download - see trap below
raw_en = datasets.load_dataset("UCSD26/medical_dialog", name="en", revision=REV,
                                data_dir="/path/to/Medical-Dialogue-Dataset-English", trust_remote_code=True)
```

**Trap**: for the `en` and `zh` configs, `_split_generators` in the loading script raises `FileNotFoundError` unless a `data_dir` pointing at a manually downloaded and unzipped folder is supplied - the script's `manual_download_instructions` sends you to a Google Drive folder to download and unzip by hand first, and warns that zipping can silently drop files over 500 MB, so single-file downloads are recommended [3]. Only `processed.en` and `processed.zh` download automatically, via direct Google Drive file links embedded in the script [3]; this was confirmed live by fetching the `processed.en` Google Drive URL directly, which returned exactly the 482 train rows the card declares [7].

## Neighbors

- `bigbio/meddialog` - a BigBio-project repackaging of the same English/Chinese data under the same citation, also a code-executing loading script with the Hub viewer disabled ("this dataset runs arbitrary python code") [7]. Same underlying content as this release, different maintainer.
- `lighteval/med_dialog` - a Parquet reformatting of only the English side, split by source site into `healthcaremagic` (181,122/22,641/22,642 train/validation/test) and `icliniq` (24,851/3,105/3,108) configs, with a working Hub viewer [8]. A fetched row shows `src` (the full dialogue) paired with a short `tgt` string that reads as a one-line patient query/summary, not the raw turn-by-turn transcript this release serves - a different task shape (summarization/query-generation pairs), not a drop-in replacement [8]. Prefer this neighbor only when a live-viewable, split-by-site English loader is what's needed; prefer this release when the raw multi-turn `dialogue_turns` structure or the Chinese data is required.
- `lavita/ChatDoctor-HealthCareMagic-100k` - a same-domain, differently-shaped (`instruction`/`input`/`output`) 112,165-row release; its own card states no provenance information ("More Information needed"), so no lineage claim to this release is made here [9].

## A row

Two distinct served shapes. The `en` and `zh` raw configs share one schema (`file_name`, `dialogue_id`, `dialogue_url`, `dialogue_turns`) differing only in the `speaker` label set, so one example covers both; `en` itself could not be fetched live because it requires the manual Google Drive download described above, so the card's own quoted Chinese example is used [2]. `processed.en` and `processed.zh` differ (only `processed.en` carries `description`), and `processed.en` was fetched live via the loading script's own Google Drive URL; `processed.zh` is 2.1 GB (`download_size`) and was not downloaded for this card, so its example is the card's own quoted row [2][4].

Raw shape, config `zh`, `dialogue_id=2` (from the dataset card's own example) [2]:

```json
{
  "dialogue_id": 2,
  "dialogue_turns": [
    {"speaker": "病人", "utterance": "孩子哭闹时，鸡鸡旁边会肿起，情绪平静时肿块会消失，去一个私人诊所看过，说是疝气.如果确定是疝气，是不是一定要手术治疗？我孩子只有1岁10月，自愈的可能性大吗？如果一定要手术，这么小的孩子风险大吗？术后的恢复困难吗？谢谢."},
    {"speaker": "医生", "utterance": "南方医的B超说得不清楚，可能是鞘膜积液，可到我医院复查一个B超。"}
  ],
  "dialogue_url": "https://www.haodf.com/doctorteam/flow_team_6477251152.htm",
  "file_name": "2020.txt"
}
```

`processed.en`, `split="train"`, row 0, fetched live from the script's own Google Drive source file (414,490 bytes, 482 rows, matching the card's declared count exactly) [7]:

```json
{
  "description": "throat a bit sore and want to get a good imune booster, especially in light of the virus. please advise. have not been in contact with nyone with the virus.",
  "utterances": [
    "patient: throat a bit sore and want to get a good imune booster, especially in light of the virus. please advise. have not been in contact with nyone with the virus.",
    "doctor: during this pandemic. throat pain can be from a strep throat infection (antibiotics needed), a cold or influenza or other virus, or from some other cause such as allergies or irritants. usually, a person sees the doctor (call first) if the sore throat is bothersome, recurrent, or doesn't go away quickly. covid-19 infections tend to have cough, whereas strep throat usually lacks cough but has more throat pain. (3/21/20)"
  ]
}
```

`processed.zh` (no `description` field, from the dataset card's own example) [2]:

```json
{
  "utterances": [
    "病人：强制性脊柱炎，晚上睡觉翻身时腰骶骨区域疼痛，其他身体任何部位均不疼痛。",
    "医生：应该没有问题，但最好把图像上传看看。"
  ]
}
```

## Where it came from

Built and released by UCSD (author `UCSD26` on the Hub); the point of contact named on the card is Pengtao Xie [2]. English dialogues are scraped from healthcaremagic.com and icliniq.com (the card also names healthtap.com as a source for the `en` config's structure description); Chinese dialogues are scraped from haodf.com [2]. The card states copyright of the underlying content belongs to those sites, and quotes both icliniq.com's and healthcaremagic.com's terms of service prohibiting scraping, republishing, duplicating, or reselling the content "for commercial or any other purpose whatsoever" [2]. The `en` and `zh` configs are the raw scraped consultations, parsed by the loading script's line-heuristic parser into `(file_name, dialogue_id, dialogue_url, dialogue_turns)` records; `processed.en` and `processed.zh` are a separately hosted, already-split (train/validation/test) reformatting into `description` + `utterances` [2][3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the `datasets-server` endpoints used here take no revision parameter and report the current, unpinned state of the Hub viewer.

[1] Chen et al., "MedDialog: Two Large-scale Medical Dialogue Datasets", 2020. https://arxiv.org/abs/2004.03329 - current title read from the live abs page. Fetched 2026-08-11.

[2] UCSD26/medical_dialog dataset card (README). https://huggingface.co/datasets/UCSD26/medical_dialog/raw/main/README.md - config descriptions, example rows, split counts, source-site ToS quotes, curation/annotation sections. Fetched 2026-08-11.

[3] `medical_dialog.py` loading script. https://huggingface.co/datasets/UCSD26/medical_dialog/raw/main/medical_dialog.py - manual-download requirement and instructions for `en`/`zh`, automatic Google Drive URLs for `processed.en`/`processed.zh`, and the turn-parsing logic. Fetched 2026-08-11.

[4] Hugging Face Hub API record for UCSD26/medical_dialog. https://huggingface.co/api/datasets/UCSD26/medical_dialog?full=true - `sha`, `cardData.dataset_info` (per-config row/byte counts), `cardData.viewer`, license tag, gated/private status, downloads, likes, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] Semantic Scholar Graph API, citations of the origin paper [1]. https://api.semanticscholar.org/graph/v1/paper/arXiv:2004.03329/citations?fields=title,year,citationCount&limit=50 - returned the 50 most-recent citing-paper titles (response carries a `next` cursor, so more citing papers exist beyond these 50); titles only were checked, not full text. Fetched 2026-08-11.

[6] datasets-server info/size/splits endpoints. https://datasets-server.huggingface.co/info?dataset=UCSD26%2Fmedical_dialog , https://datasets-server.huggingface.co/size?dataset=UCSD26%2Fmedical_dialog , https://datasets-server.huggingface.co/splits?dataset=UCSD26%2Fmedical_dialog - each returns HTTP 501, "dataset viewer is disabled". Fetched 2026-08-11.

[7] `bigbio/meddialog` dataset card (README) and its datasets-server `/info` response (which returns "this dataset runs arbitrary python code"). https://huggingface.co/datasets/bigbio/meddialog/raw/main/README.md , https://datasets-server.huggingface.co/info?dataset=bigbio%2Fmeddialog - neighbor comparison; also covers the live fetch of the `processed.en` Google Drive source file (`https://drive.google.com/uc?export=download&id=1ria4E6IdTIPsikL4Glm3uy1tFKJKw0W8`, referenced by [3]), read directly as JSON, 482 rows. Fetched 2026-08-11.

[8] `lighteval/med_dialog` dataset card, datasets-server `/info`, and a fetched first row. https://huggingface.co/datasets/lighteval/med_dialog/raw/main/README.md , https://datasets-server.huggingface.co/info?dataset=lighteval%2Fmed_dialog , https://datasets-server.huggingface.co/first-rows?dataset=lighteval%2Fmed_dialog&config=healthcaremagic&split=train - neighbor comparison. Fetched 2026-08-11.

[9] `lavita/ChatDoctor-HealthCareMagic-100k` dataset card (README). https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/raw/main/README.md - schema and row count; card states no provenance information. Fetched 2026-08-11.

[10] The corpus screening row for `UCSD26/medical_dialog`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as research-scoped dialogue data, with two conditions the card already establishes: the raw `en`/`zh` configs need a manual Google Drive download before `load_dataset` will run, and the whole release sits under an unmarked license plus source-site terms of service that bar scraping/republication [2][3][4]. The screening row's note names the corpus, its languages, and the scrape origin, and the viewer's HTTP 501 as the reason the row/size/splits fields on the shortlist are empty [10].

### The screening row

The row's own note [10]: "The MedDialog corpus of doctor-patient conversations in English and Chinese, scraped from healthcaremagic.com, icliniq.com and Chinese consultation sites; the card declares 5,558,895 rows, and because the repo is a loading script the viewer returns HTTP 501." The row carries no flag.
