# dennlinger/eur-lex-sum

EUR-Lex-Sum: long EU legal acts paired with human-written summaries in up to 24 languages - 1,129 / 187 / 188 English documents - for long-form legal summarisation.

**dennlinger/eur-lex-sum** is "a multilingual resource intended for text summarization in the legal domain", "based on human-written summaries of legal acts issued by the European Union" [1], introduced by Aumiller, Chouhan and Gertz [2]. Validation and test documents are available in all 24 languages and aligned at the paragraph level; training sets differ by language [1]. It lives at https://huggingface.co/datasets/dennlinger/eur-lex-sum .

**Use it for**: SFT for long-document summarisation of legislation, in English or cross-lingually. Small in English (1,129 training documents) but with long, high-quality references.

**Licence**: CC BY 4.0 in the card metadata [3], while the card's Licensing Information says "Data from the EUR-Lex platform is available under the CC-BY SA 4.0 license. We redistribute the dataset under the same license" [1]. The one catch: metadata and body disagree, the opposite way round from MultiEURLEX.

**Shape**: English: `train` 1,129 / `validation` 187 / `test` 188, counted from the downloaded files [4]; three fields (`celex_id`, `reference`, `summary`) [1]. The viewer does not run the script [5].

**Hold out**: `validation` and `test` (375 acts available in all 24 languages, split 187 / 188) [1]. The card says the authors "ensured that no duplicates exist across the three splits" by exact match on reference and summary [1]. `lighteval/legal_summarization`'s `EurLexSum` config has the same English counts [6].

**Origin**: EU legal acts and the EU's own human-written summaries [1]. Hub API at the check date: `downloads` 1,783, `downloadsAllTime` 39,412, `likes` 51 [3].

**Trained-on-by**: the Hub's dataset tag returns 20 models, the query's limit, mostly summarisation fine-tunes [7].

**Introduced by**: [2] (Aumiller et al.).

## Shape

The viewer serves no rows [5]. English rows counted from `data/english/{train,validation,test}.json` [4]:

| split | rows |
| --- | --- |
| `train` | 1,129 |
| `validation` | 187 |
| `test` | 188 |

Each of the 24 languages has its own directory under `data/` with the three JSON files [8].

## Quality

- The card reports that validation and test are paragraph-aligned across all 24 languages [1], which makes cross-lingual evaluation exact.

## Load it

Load one language by config name, pinned:

```python
import datasets

REV = "33ecb2d630298e3f912d067aaaa71aaf4ee92404"  # main at the check date
train = datasets.load_dataset("dennlinger/eur-lex-sum", "english", revision=REV, split="train", trust_remote_code=True)  # 1,129
test = datasets.load_dataset("dennlinger/eur-lex-sum", "english", revision=REV, split="test", trust_remote_code=True)    # 188 - hold out
```

**Trap**: config names are full language names in lower case (`english`, `german`), matching the `data/<language>/` directories, not ISO codes [8][9].

## Neighbors

- `coastalcph/multi_eurlex` - EU laws with topic labels, not summaries.
- `lighteval/legal_summarization` - includes this dataset's English split as `EurLexSum`.

## A row

The viewer serves no rows. The card's documented instance [1]:

```json
{
  "celex_id": "3A32021R0847",
  "reference": "REGULATION (EU) 2021/847 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL\n [...]",
  "summary": "Supporting EU cooperation in the field of taxation: Fiscalis (2021-2027)\n\n [...]"
}
```

## Where it came from

Built by Dennis Aumiller, Ashish Chouhan and Michael Gertz from EUR-Lex; code at https://github.com/achouhan93/eur-lex-sum [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] dennlinger/eur-lex-sum dataset card (README). https://huggingface.co/datasets/dennlinger/eur-lex-sum/raw/main/README.md. Fetched 2026-09-23.

[2] Aumiller et al., "EUR-Lex-Sum: A Multi- and Cross-lingual Dataset for Long-form Summarization in the Legal Domain", arXiv:2210.13448, 2022. https://arxiv.org/abs/2210.13448 - current title read from the live abs page. Fetched 2026-09-23.

[3] Hugging Face Hub API record for dennlinger/eur-lex-sum. https://huggingface.co/api/datasets/dennlinger/eur-lex-sum?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] dennlinger/eur-lex-sum files `data/english/train.json`, `validation.json`, `test.json`, downloaded and counted by this skill. Fetched 2026-09-23.

[5] datasets-server size, info and splits endpoints for dennlinger/eur-lex-sum. https://datasets-server.huggingface.co/size?dataset=dennlinger%2Feur-lex-sum - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[6] datasets-server size endpoint for lighteval/legal_summarization. https://datasets-server.huggingface.co/size?dataset=lighteval%2Flegal_summarization - `EurLexSum` 1,129 / 187 / 188. Fetched 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:dennlinger/eur-lex-sum&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] Repository file tree for dennlinger/eur-lex-sum. https://huggingface.co/api/datasets/dennlinger/eur-lex-sum/tree/main?recursive=true. Fetched 2026-09-23.

[9] dennlinger/eur-lex-sum repository file `eur-lex-sum.py`. https://huggingface.co/datasets/dennlinger/eur-lex-sum/raw/main/eur-lex-sum.py. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for long-form legal summarisation with human references. Record the CC BY 4.0 (metadata) versus CC BY-SA 4.0 (body) discrepancy.

### The screening row

The row's own note: "EUR-Lex-Sum; human summaries; licence metadata vs body mismatch." The row carries the flag `script-loaded`.
