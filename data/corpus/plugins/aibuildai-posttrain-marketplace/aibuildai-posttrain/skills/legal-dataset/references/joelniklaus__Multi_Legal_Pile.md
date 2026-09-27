# joelniklaus/Multi_Legal_Pile

A 689GB multilingual legal pretraining corpus in 24 languages - case law, legislation, contracts and other legal text from EU and national sources - assembled from four collections at load time by a loading script.

**joelniklaus/Multi_Legal_Pile** was introduced by Niklaus et al. in "MultiLegalPile: A 689GB Multilingual Legal Corpus" [1]. The card describes it as "a large-scale multilingual legal dataset suited for pretraining language models" spanning "over 24 languages and five legal text types" (`caselaw`, `contracts`, `legislation`, `other`, `legal_mc4`) [2]. The 689GB total is four collections: native Multi Legal Pile data (112GB), Eurlex Resources (179GB), Legal MC4 (106GB) and Pile of Law (292GB) [2]. It lives at https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile .

**Use it for**: continued pretraining for European legal language, one `{language}_{type}` config at a time (for example `de_caselaw` or `fr_legislation`) [2]. It is the only large corpus here with native non-English European case law and legislation. Not SFT data.

**Licence**: CC BY-NC-SA 4.0 in the card metadata [3], but the card's Licensing Information section reads "[More Information Needed]" [2], and its native-data table lists a per-source licence for each file (for example CC0-1.0 for Bulgarian legislation and CC BY-NC 4.0 for Czech Constitutional Court decisions) [2]. The card says `legal_mc4` is listed separately "so it can be easily excluded since it is less permissively licensed than the other types" [2]. The one catch: the effective licence is the most restrictive of the sources a config pulls.

**Shape**: a `{language}_{type}` config for each of 24 languages plus `all`, times five types plus `all` [4]; one `train` split per config [2]. The dataset viewer does not run the loading script, so no row count is served [5].

**Hold out**: nothing inside the repository - one `train` split. The English configs pull Pile of Law, which its authors did not decontaminate, so the same screening applies; see the `pile-of-law/pile-of-law` card.

**Origin**: documents written by courts and legislatures across the EU, Switzerland, Brazil and elsewhere, plus web text classified as legal (Legal MC4) [2]. Hub API at the check date: `downloads` 10,606, `downloadsAllTime` 97,912, `likes` 67 [3].

**Trained-on-by**: the Hub's dataset tag lists the `morenolq/LEGIT-BART` family of Italian legal BART models (8-20 downloads each) [6]. The paper trains the authors' own multilingual legal encoders on this corpus [1].

**Introduced by**: [1] (Niklaus et al.).

## Shape

The viewer answers "The dataset viewer doesn't support this dataset because it runs arbitrary python code" [5]. The card's own collection sizes [2]:

| collection | size | pulled from |
| --- | --- | --- |
| Native Multi Legal Pile | 112GB | this repository's `data/` |
| Eurlex Resources | 179GB | `joelito/eurlex_resources` |
| Legal MC4 | 106GB | `joelito/legal-mc4` |
| Pile of Law | 292GB | `pile-of-law/pile-of-law` (English configs only) |

The script's `_split_generators` downloads from three other repositories unless `repo_data_only` is set: Eurlex Resources for every language, Legal MC4 for the `legal_mc4` type, and Pile of Law for every English config [4]. This repository's own `data/` tree holds only the native files, 33 `.jsonl.xz` files across 14 languages [7].

## Quality

- Most of the card's standard sections - Data Splits, Curation Rationale, Personal and Sensitive Information, Licensing Information - read "[More Information Needed]" [2].
- Legal MC4 is web text filtered for legal content, not text published by a legal authority [2]; exclude the `legal_mc4` type when provenance matters.
- No source states a measured duplicate rate. Eurlex Resources and several native sources both cover EU law, so overlap across collections is likely and unmeasured.

## Load it

The script must run, and it fetches from other repositories. Stream one config and pin this repository's revision:

```python
import datasets

REV = "911e1d214162fd11d2c78d3f1428cbfcbe07782c"  # main at the check date
de_caselaw = datasets.load_dataset("joelniklaus/Multi_Legal_Pile", "de_caselaw", revision=REV,
                                   split="train", streaming=True, trust_remote_code=True)
```

**Trap**: the English configs download Pile of Law's `train` *and* `validation` files and yield both inside a single `train` split [4], so the 25% Pile of Law set aside as validation is inside this corpus's training data. The revision pin covers only this repository: the script also downloads from `joelito/Multi_Legal_Pile`, `joelito/eurlex_resources` and `joelito/legal-mc4` by name, unpinned [4], so a load can change when those repositories change.

## Neighbors

- `pile-of-law/pile-of-law` - the English portion, loaded directly and with its own validation split kept separate.
- `joelniklaus/eurlex_resources` and `joelniklaus/legal-mc4` - two of the component collections, visible in the Hub search under `joelniklaus/` [8].
- `coastalcph/multi_eurlex` - EU legislation labelled with EuroVoc concepts in 23 languages: task data rather than pretraining text.

## A row

The viewer serves no rows for this repository [5]. The card's documented row shape, per its Data Fields description in the loading script, is [4]:

```json
{
  "language": "de",
  "type": "caselaw",
  "jurisdiction": "Germany",
  "text": "<document text>"
}
```

This is the field layout the script declares, not a fetched row: no row was read for this card.

## Where it came from

Assembled by Joel Niklaus, Veton Matoshi, Matthias Stürmer, Ilias Chalkidis and Daniel E. Ho [1][2]. The native portion comes from national court and legislation portals and the MARCELL corpora, each listed with its URL and licence in the card's table [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Niklaus et al., "MultiLegalPile: A 689GB Multilingual Legal Corpus", arXiv:2306.02069, 2023. https://arxiv.org/abs/2306.02069 - current title read from the live abs page. Fetched 2026-09-23.

[2] joelniklaus/Multi_Legal_Pile dataset card (README). https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile/raw/main/README.md. Fetched 2026-09-23.

[3] Hugging Face Hub API record for joelniklaus/Multi_Legal_Pile. https://huggingface.co/api/datasets/joelniklaus/Multi_Legal_Pile?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] joelniklaus/Multi_Legal_Pile repository file `Multi_Legal_Pile.py`. https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile/raw/main/Multi_Legal_Pile.py - `_split_generators`, the `download_url` calls into `joelito/*` and `pile-of-law/pile-of-law`, and the config list. Fetched 2026-09-23.

[5] datasets-server size, info and splits endpoints for joelniklaus/Multi_Legal_Pile. https://datasets-server.huggingface.co/size?dataset=joelniklaus%2FMulti_Legal_Pile - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:joelniklaus/Multi_Legal_Pile&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] Repository file tree for joelniklaus/Multi_Legal_Pile. https://huggingface.co/api/datasets/joelniklaus/Multi_Legal_Pile/tree/main?recursive=true. Fetched 2026-09-23.

[8] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=legal&sort=downloads and https://huggingface.co/api/datasets?search=eurlex&sort=downloads - list `joelniklaus/legal-mc4` and `joelniklaus/eurlex_resources`. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as non-commercial continued-pretraining text for European legal language. Not auditable through the viewer; the English configs inherit Pile of Law's contamination risk and fold its validation split into training.

### The screening row

The row's own note: "multilingual legal pretraining corpus; script-loaded, pulls three other repos, English configs = Pile of Law train+validation." The row carries the flag `script-loaded`.
