# FredrikBL/Legal-Snigel-DPO

1,998 synthetic Swedish DPO pairs over Swedish statutes: the prompt is the text of an ordinance, and chosen and rejected are two plain-language summaries of it.

**FredrikBL/Legal-Snigel-DPO** has a card consisting only of front matter: tags `dpo` and `synthetic`, language `sv`, and `pretty_name: d` [1]. Each row names a Swedish statute by its designation (`beteckning`, for example `2013:707`) and title, gives the statute's text as `prompt`, and pairs two Swedish summaries as `chosen` and `rejected` [2]. It lives at https://huggingface.co/datasets/FredrikBL/Legal-Snigel-DPO .

**Use it for**: preference tuning for Swedish legal summarisation. The chosen and rejected summaries in row 0 differ in detail and emphasis rather than correctness [2], so the signal is subtle; read a sample before relying on it.

**Licence**: none on the card [3][1]. The one catch: no licence, and the summaries' generator is unstated.

**Shape**: 1,998 rows: `train` 1,598 / `test` 400 [4]; six columns, including a leftover `__index_level_0__` [5].

**Hold out**: `test` (400 rows). Measured: 1 of 400 `test` prompts also appears in `train` [6].

**Origin**: statute text from Swedish legislation; summaries model-generated per the `synthetic` tag [1]. Hub API at the check date: `downloads` 18, `downloadsAllTime` 696, `likes` 1 [3].

**Trained-on-by**: no model on the Hub declares this dataset through its dataset tag [7].

**Introduced by**: no paper and no card text [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 1,598 |
| `test` | 400 |
| total | 1,998 |

One config, `default` [5]:

| column | dtype |
| --- | --- |
| `beteckning` | string |
| `titel` | string |
| `prompt` | string |
| `rejected` | string |
| `chosen` | string |
| `__index_level_0__` | int64 |

## Quality

- Measured duplication: 1 `train` prompt repeats (0.06%) [8].
- Some statute texts carry expiry markers such as "/Upphör att gälla U:2024-04-23/" (ceases to apply) inside the prompt [2], so parts of the source law were already repealed when the data was made.

## Load it

Train on `train`, hold out `test`:

```python
import datasets

REV = "3aff0060c43a1e0748a9cab2030af10c86e453cc"  # main at the check date
train = datasets.load_dataset("FredrikBL/Legal-Snigel-DPO", revision=REV, split="train")  # 1,598 rows
test = datasets.load_dataset("FredrikBL/Legal-Snigel-DPO", revision=REV, split="test")    # 400 rows - hold out
train = train.remove_columns(["__index_level_0__"])
```

**Trap**: `__index_level_0__` is a pandas index left in the export [5]; some DPO trainers pass unknown columns through to the collator and fail on it. Drop it.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [2], truncated:

```json
{
  "beteckning": "2013:707",
  "titel": "Förordning (2013:707) om kontroll av vissa skjutvapen, delar till skjutvapen och ammunition",
  "prompt": "1 § I denna förordning finns föreskrifter om sådan kontroll\nav skjutvapen, delar till skjutvapen och ammunition som anges\ni Europaparlamentets och rådets förordning (EU) nr 258/2012\nav den 14 mars 2012 om genomförande av artikel 10 i FN:s\nprotokoll o [...]",
  "rejected": "Den här lagen handlar om kontroll av skjutvapen, delar till skjutvapen och ammunition. Den reglerar hur man får exportera, importera och transitera dessa varor, och vilka krav som ställs på ansökningar om tillstånd. Inspektionen för strategiska produ [...]",
  "chosen": "Den här lagen handlar om regler för kontroll av skjutvapen, delar till skjutvapen och ammunition enligt EU:s regler. Inspektionen för strategiska produkter hanterar ansökningar om exporttillstånd, och en avgift kan tas ut för detta. Ansökningar måste [...]",
  "__index_level_0__": 240
}
```

## Where it came from

Uploaded by user FredrikBL in April 2024 [3]; nothing on the repository describes how the summaries were produced [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] FredrikBL/Legal-Snigel-DPO dataset card (README). https://huggingface.co/datasets/FredrikBL/Legal-Snigel-DPO/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=FredrikBL%2FLegal-Snigel-DPO&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[3] Hugging Face Hub API record for FredrikBL/Legal-Snigel-DPO. https://huggingface.co/api/datasets/FredrikBL/Legal-Snigel-DPO?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=FredrikBL%2FLegal-Snigel-DPO - takes no revision parameter; a live figure. Fetched 2026-09-23.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=FredrikBL%2FLegal-Snigel-DPO - column schema; live, no revision parameter. Fetched 2026-09-23.

[6] This skill's own exact split-leakage count at the pinned revision - normalised `prompt`, `test` against `train`. Fetched 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:FredrikBL/Legal-Snigel-DPO&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] This skill's own duplication count at the pinned revision. Run 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as a small Swedish legal-summarisation preference set after a read-through; no licence and no stated generator.

### The screening row

The row's own note: "Swedish statute summaries, synthetic DPO; card is front matter only." The row carries the flag `no-licence`.
