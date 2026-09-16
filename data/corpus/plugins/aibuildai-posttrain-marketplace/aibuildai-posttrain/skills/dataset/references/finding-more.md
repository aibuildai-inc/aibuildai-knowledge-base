# Finding a dataset that is not on the list

This file teaches an action instead of stating facts, because the list is a snapshot taken on 2026-08-11 and datasets appear, change, and disappear. When `index.md` holds nothing that fits, this is how to go and look, and how to judge what you find.

## Order to search in

Cheapest and most likely first.

1. **The Hub, by name and by author.** `GET https://huggingface.co/api/datasets?search=<name>&author=<owner>&full=true`. Query the compound name, not a generic word: this endpoint tokenizes the id, so a tail word usually misses. Measured on eight well-known datasets, a tail word found 2 of 8 and the compound name found 8 of 8. A missing repository answers 401 anonymously and 404 with a token, so never key a decision on one status code.
2. **The Hub, by full text.** `GET https://huggingface.co/api/search/full-text?q=<phrase>&type=dataset`. This is the only call that can support the sentence "it is not on the Hub", because it reads card text and not only ids. It caps `skip` at 1000 without saying so: HTTP 200, empty hits, unchanged total.
3. **Paper to repository.** `GET https://huggingface.co/api/arxiv/<arxiv-id>/repos` turns a paper id into the repositories that cite it. It is a citation index and not a release index - it found 11 of 14 known releases - so pair it with an author search and a name search.
4. **The DOI repositories.** Zenodo, Harvard Dataverse and its many separate installations, figshare, OSF, Dryad, Mendeley. This is a tier, not one site: every one of them contributed only datasets no other channel had, and Zenodo is one member of the group rather than a superset of it. Two limits to know: the `dataverse.unc.edu` installation answers 401 to anonymous search, and Mendeley is unreached rather than empty, because every endpoint answers 403 behind a bot page.
5. **GitHub.** Read a repository's file list with `GET /repos/<owner>/<repo>/git/trees/<sha>?recursive=1`, and follow large-file pointers to the real bytes on `raw.githubusercontent.com`. This is what says where the data really came from.
6. **The paper's own repository README.** The only route to a release hosted on a drive or a university server. Grep the README for release links.

## Traps that return a wrong answer without failing

Each one answers HTTP 200 with something believable.

1. **The plain `?search=` call is not a test for absence.** It matches repository ids, not card text. Measured: `IndicPersonaHub` gives 1 hit by search and 0 by full text, while `geophysics preference` gives 0 by search and 116 by full text. Only the full-text call can support an absence claim.
2. **`?search=` is tokenized, so a generic word will not match inside a compound name.** `?search=hermes` returns 510 rows and `teknium/OpenHermes-2.5` is in none of them; `?search=OpenHermes` returns 498 rows with it first. The same holds for chat against ultrachat, feedback against ultrafeedback, instruct against OpenMathInstruct, rlhf against SafeRLHF, and steer against HelpSteer. A keyword list of generic words silently misses the best-known dataset in every family.
3. **The GitHub directory listing stops at 1000 entries with no signal.** No link header, no truncation flag, and `per_page` and `page` are ignored. Measured on `awslabs/open-data-registry` at commit `628dc31395bd96f2ec2b3424af963ffbde5f48ce`: the contents call gives 1000 files, the recursive tree call gives 1176 and reports its own `truncated` flag as false. Use the tree call and read that flag.
4. **A multi-word query on a DOI repository is joined with OR unless it is quoted.** Measured on Dataverse: unquoted, `preference dataset` returns 301,709 records; quoted, it returns 7. Quote every phrase.
5. **Zenodo needs URL encoding, and it fails silently without it.** Raw spaces make the request exit with zero bytes. Use `curl -s -G "https://zenodo.org/api/records" --data-urlencode 'q=resource_type.type:dataset AND (title:"instruction tuning" OR title:"SFT")' -d size=25 -d sort=newest`. Without a login the page size is 25; `size=50` answers 400.
6. **A download link has to be checked at the last hop.** A file host can answer a redirect chain that ends in a small warning page, and a naive download saves that page under the file name you asked for: a Google Drive id that looks alive returns a 2425-byte virus-scan warning page. Only a `content-length` plus a `content-disposition: attachment` header on the final response proves real bytes.

## Rate limits and credentials

| Call | Limit | Token |
|---|---|---|
| Hugging Face API (search, full text, tree, datasets-server) | 500 requests per 300 seconds, cursor paging, no total-count header | not needed; a token changes which status code a missing repository returns |
| GitHub REST (repos, trees, contents) | 60 requests per hour anonymous, 5000 with a token | strongly wanted; without one, screening runs at about 12 datasets an hour |
| GitHub code search | 10 requests per minute, hard 403 on the eleventh | required: the endpoint answers 401 without one |
| Zenodo | 30 requests per minute, 25 records per page | not needed |
| Dataverse installations | not published | not needed, except `dataverse.unc.edu`, which answers 401 anonymously |

## The quality gate to run on what you find

A search result is not a verdict. Run this before using anything, in this order, and stop at the first refusal.

| Stage | What it asks | How it is answered |
|---|---|---|
| 0 can it be read | is it gated, and does the metadata endpoint answer | three refusals: `gated` is not false, `/info` answers 501 (the repository runs its own loading code), or `/info` answers 500 (no supported data files). Then read `partial` from `/size`. A gated repository can still be worth a tree read: one measured case answers 200 on the repository API with or without a token while the datasets-server answers 404 |
| 1 what kind is it | the column shape | read both chat dialects - `messages` as a list of role/content structs, and `conversations` as a list of from/value structs - plus `eval.yaml` at the repository root. That file is precise and not sensitive: it fired on 3 of 7 known evaluation sets with no false alarms, so use it to confirm an evaluation set, never to rule one out. Split names tell you nothing: benchmark repositories carry `train`, and training corpora carry `test` |
| 2 licence | does the card's field match its own body | compare `cardData.license` against the card text, and put model names in the pattern, because a card writes "GPT-4" and never "OpenAI". Then follow `cardData.source_datasets` one step upstream: a permissively licensed release can be built entirely from restricted output |
| 3 duplication | how much of it is the same row twice | one Unicode-aware cleaner first, then exact, cleaned, and MinHash rates. An ASCII-only character class empties every non-Latin row and invents duplicate groups: on one 939,343-row corpus it reported 14.04% where the truth is 8.51%. The 200-character-prefix shortcut is banned - it does not even keep the order of two datasets right |
| 4 contamination | does it carry the evaluation's own items | three controls, or the zeros mean nothing: the evaluation against itself (1319 of 1319), the same text with its words sorted (0 of 1319), and a positive control known to contain the evaluation's training split. Match against the whole row text, not the prompt alone |
| 5 overlap with other datasets | is the declared source list honest | join it against the datasets you already trust. Measured: one corpus contains a second entirely, as its card says, while another is about 22% inside a third and nobody says so |
| 6 and 7 origin and shape | who wrote it, and what it looks like | the `/statistics` frequencies endpoint, free and exact, but only when `partial` is false |

If `partial` is true, no number from stages 3 to 7 may be reported as a whole-split figure. Mark the dataset as not auditable and take its row count from its card, or stream the repository's own files with range reads against `/api/datasets/<id>/tree/main?recursive=true&expand=true`. Also read the file list from that tree even when the dataset is not partial: the `/info` endpoint leaves configs out, and a measured case hid a 56,339-row preference file that no datasets-server answer mentions at all.

Text markers cannot recover who wrote the text: human-written data scored 4.50% on style markers and machine-generated data scored 5.17%. The licence route is stronger. One corpus passes a licence check as clean MIT while its upstream card names GPT-4 eight times and 1.31% of its chosen answers contain "as an ai language model".

## After you find one

Write a card in the shape the other cards in this skill use, and give it the same appendix: what the gate asked and what it answered. A dataset found by search is not exempt from the gate.
