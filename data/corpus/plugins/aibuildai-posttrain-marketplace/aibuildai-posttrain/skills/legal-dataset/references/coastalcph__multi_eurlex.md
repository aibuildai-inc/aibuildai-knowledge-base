# coastalcph/multi_eurlex

MultiEURLEX: 65,000 EU laws in up to 23 languages, each labelled with EuroVoc concepts at three levels of granularity - multilingual multi-label classification with parallel splits.

**coastalcph/multi_eurlex** is MultiEURLEX, "A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer" by Chalkidis, Fergadiotis and Androutsopoulos [1]. Each document is an EU law identified by its `celex_id`, its text in one or all languages, and EuroVoc concept labels [2]. It lives at https://huggingface.co/datasets/coastalcph/multi_eurlex .

**Use it for**: SFT for multilingual legal topic classification, or cross-lingual transfer experiments: train in one language, test in another. Its texts are also a clean parallel corpus of EU legislation.

**Licence**: CC BY-SA 4.0 in the card metadata [3], while the card's Licensing Information says "We provide MultiEURLEX with the same licensing as the original EU data (CC-BY-4.0)" [2]. The one catch: the metadata and the body name different licences; the body's reasoning (EUR-Lex reuse policy) supports CC BY 4.0.

**Shape**: per language 55,000 / 5,000 / 5,000 documents (`train` / `validation` / `test`) for the largest languages, fewer training documents for later-joining member states (Polish 23,197; Romanian 15,921) [2]. Configs are languages plus `all_languages` [4]. The viewer does not run the script [5].

**Hold out**: `test` in every language; the splits are parallel across languages (the same laws), so never train on one language's `test` and evaluate on another's.

**Origin**: EU legislation from EUR-Lex, originally annotated with EuroVoc concepts at levels 3 to 8 [2]. Hub API at the check date: `downloads` 1,828, `downloadsAllTime` 182,201, `likes` 46 [3].

**Trained-on-by**: the Hub's dataset tag lists one model [6]. LexGLUE's `eurlex` task uses the English portion [7].

**Introduced by**: [1] (Chalkidis et al.).

## Shape

The viewer serves no rows [5]. Per-language document counts, from the card [2]:

| language | `train` / `validation` / `test` |
| --- | --- |
| en, de, fr, it | 55,000 / 5,000 / 5,000 |
| es | 52,785 / 5,000 / 5,000 |
| pl | 23,197 / 5,000 / 5,000 |
| ro | 15,921 / 5,000 / 5,000 |

All data is one 2,770,050,147-byte archive, `data/multi_eurlex.tar.gz`, downloaded whole by the script regardless of which language is requested [8][4].

## Quality

- Four label sets per document: EuroVoc level 1, level 2, level 3, and the original sparse assignment; the card warns levels 4-8 "cannot be used independently" [2]. Choose the level deliberately.
- The split is chronological - training 1958-2010, development 2010-2012, test 2012-2016 - so test scores include concept drift [2].

## Load it

Load one language; the script downloads the whole 2.8 GB archive:

```python
import datasets

REV = "2020d0350241461069a54177b639f0e6c7a7a712"  # main at the check date
en = datasets.load_dataset("coastalcph/multi_eurlex", "en", revision=REV, split="train", trust_remote_code=True)  # 55,000 docs
```

**Trap**: the card's own snippet is `load_dataset('multi_eurlex', 'all_languages')` [2] - the old canonical name, not this repository's id. Use `coastalcph/multi_eurlex`.

## Neighbors

- `dennlinger/eur-lex-sum` - human-written summaries of EU legal acts in 24 languages.
- `coastalcph/lex_glue` `eurlex` - the English classification task.

## A row

The viewer serves no rows. The card's documented instance, abridged [2]:

```json
{
  "celex_id": "31979D0509",
  "text": {"en": "COUNCIL DECISION  of 24 May 1979  on financial aid from the Community for the eradication of African swine fever in Spain  (79/509/EEC) [...]", "...": "..."},
  "labels": [...]
}
```

Abridged from the card's example, not fetched.

## Where it came from

Built by Ilias Chalkidis, Manos Fergadiotis and Ion Androutsopoulos from EUR-Lex; code at https://github.com/nlpaueb/multi-eurlex [2][1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Chalkidis et al., "MultiEURLEX -- A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer", arXiv:2109.00904, 2021. https://arxiv.org/abs/2109.00904 - current title read from the live abs page. Fetched 2026-09-23.

[2] coastalcph/multi_eurlex dataset card (README). https://huggingface.co/datasets/coastalcph/multi_eurlex/raw/main/README.md. Fetched 2026-09-23.

[3] Hugging Face Hub API record for coastalcph/multi_eurlex. https://huggingface.co/api/datasets/coastalcph/multi_eurlex?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] coastalcph/multi_eurlex repository file `multi_eurlex.py`. https://huggingface.co/datasets/coastalcph/multi_eurlex/raw/main/multi_eurlex.py. Fetched 2026-09-23.

[5] datasets-server size, info and splits endpoints for coastalcph/multi_eurlex. https://datasets-server.huggingface.co/size?dataset=coastalcph%2Fmulti_eurlex - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:coastalcph/multi_eurlex&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] coastalcph/lex_glue dataset card (README), Source Data table. https://huggingface.co/datasets/coastalcph/lex_glue/raw/main/README.md. Fetched 2026-09-23.

[8] Repository file tree for coastalcph/multi_eurlex. https://huggingface.co/api/datasets/coastalcph/multi_eurlex/tree/main?recursive=true. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for multilingual legal classification. Metadata says CC BY-SA 4.0, body says CC BY 4.0; follow the body's EUR-Lex basis and record the discrepancy.

### The screening row

The row's own note: "MultiEURLEX; script-loaded 2.8 GB archive; licence metadata vs body mismatch." The row carries the flag `script-loaded`.
