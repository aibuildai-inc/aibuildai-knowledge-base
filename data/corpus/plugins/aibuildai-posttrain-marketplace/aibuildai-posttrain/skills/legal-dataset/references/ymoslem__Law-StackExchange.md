# ymoslem/Law-StackExchange

Every question and answer on the Law Stack Exchange site up to 14 August 2023 - 24,370 questions with their scored answers as HTML, under CC BY-SA - the largest human-written legal Q&A set in English here.

**ymoslem/Law-StackExchange** contains "All StackExchange legal questions and their answers from the Law site, up to 14 August 2023", collected through the official StackExchange API with the notebook included in the repository [1]. Each row is one question with its title, body, tags, score and a list of answers, each answer carrying its own score [2]. It lives at https://huggingface.co/datasets/ymoslem/Law-StackExchange .

**Use it for**: SFT on lay legal questions answered by people, choosing the top-scored answer per question as the target, or preference pairs built from answer scores. Questions span many jurisdictions and are often hypothetical, which suits general legal-assistant behaviour better than jurisdiction-specific practice.

**Licence**: CC BY-SA 4.0 in the card metadata [3]; each row also carries the post's own licence, and the sampled rows mix "CC BY-SA 4.0" (35) and "CC BY-SA 3.0" (2) [4]. The one catch: share-alike - a model's outputs are not clearly covered, but redistributing the data or derivatives requires the same licence and attribution.

**Shape**: 24,370 rows in one `train` split [5]; eight columns, `answers` a list of `{answer_id, body, score}` structs [2].

**Hold out**: no split is set aside. No benchmark in this skill is drawn from Law Stack Exchange; `jonathanli/law-stack-exchange` is a small classification set from the same site, so remove its questions before evaluating on it.

**Origin**: questions and answers written by users of law.stackexchange.com [1]. Hub API at the check date: `downloads` 154, `downloadsAllTime` 9,367, `likes` 32 [3].

**Trained-on-by**: the Hub's dataset tag lists `werty1248/Mistral-Nemo-NT-Ko-12B-sft` and its GGUF quantisations (the i1 GGUF at 2,519 downloads), plus legal embedding models such as `bugBug04S/legal-embed-modernbert-v2` [6].

**Introduced by**: no paper - the dataset card, which carries a DOI, 10.57967/hf/4792 [3][1].

## Shape

| split | rows |
| --- | --- |
| `train` | 24,370 |
| total | 24,370 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `question_title` | string |
| `score` | int64 |
| `link` | string |
| `license` | string |
| `question_body` | string |
| `question_id` | int64 |
| `answers` | list<struct<answer_id: int64, body: string, score: int64>> |
| `tags` | list<string> |

## Quality

- Bodies are raw HTML: all 37 sampled rows contain `<p>` tags in `question_body` [4]. Strip or convert to Markdown before training, or the model learns to emit HTML.
- Answer counts vary: of 37 sampled questions, 16 have one answer, 11 three, 9 two and 1 nine [4]. Questions with no answer would give no target; none appeared in the sample.
- Measured: all 24,370 `question_id` values are unique [7].
- Answers are community-written and scored, not verified; a high score is agreement, not correctness.

## Load it

One split; pin the revision:

```python
import datasets

REV = "ab2dbaad9a71b6550316994c8fcd69f24f7f16e6"  # main at the check date
lse = datasets.load_dataset("ymoslem/Law-StackExchange", revision=REV, split="train")  # 24,370 rows
def best(row):
    a = max(row["answers"], key=lambda x: x["score"]) if row["answers"] else None
    return {"prompt": row["question_title"] + "\n\n" + row["question_body"], "response": a["body"] if a else None}
```

**Trap**: the repository's data file is one JSON array, `law-stackexchange-questions-answers.json`, not Parquet [8]; the viewer converts it, but tools that read the raw file must parse a 106 MB single JSON document, not JSONL.

## Neighbors

- `jonathanli/law-stack-exchange` - 2,553 Law Stack Exchange questions without answers, labelled by topic for classification (`train` 638 / `validation` 319 / `test` 1,596), from "Parameter-Efficient Legal Domain Adaptation" [9].
- `dim/law_stackexchange_prompts` and `ChrisZhang312/law_stackexchange_cleaned` - derivatives found by the Hub search [10]; not screened here.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [4], with long fields truncated:

```json
{
  "question_title": "Why is drunk driving causing accident punished so much worse than just drunk driving?",
  "question_body": "<p>When people drink and drive and then cause an accident especially where if someone dies they get years and years in prison but just the act of drunk driving is punished way more lenient.  Shouldn't the 2, drunk drivin [...]",
  "answers": [
    {
      "answer_id": 94666,
      "body": "<h3>Moral luck</h3>\n<p>You have raised the issue of <em>moral luck</em>, a long recognized problem in criminal theory. The classic expositions of this issue are by <a href=\"https://en.m.wikipedia.org/wiki/Thomas_Nagel\" r [...]",
      "score": 72
    },
    {
      "answer_id": 94674,
      "body": "<p>Drunk driving remains, per se, &quot;victimless&quot; - a breach of regulations - until someone actually becomes a victim. That puts less emphasis on the punitive role of the justice system and more on deterrence and  [...]",
      "score": 26
    },
    {
      "answer_id": 94677,
      "body": "<p>Drivers are negligent all the time. Not only by drunk driving, but also by speeding, driving when really tired, etc.</p>\n<p>The question is, what level of negligence is enough to call it recklessness?</p>\n<p>The rough [...]",
      "score": 8
    },
    {
      "answer_id": 94669,
      "body": "<p>Have you seen or watched the movie Minority Report?  People were arrested and imprisoned based upon what they would have done in the future.</p>\n<p>While you are probably unable to drive in a reasonably safe manner in [...]",
      "score": 7
    },
    {
      "answer_id": 94681,
      "body": "<p><strong>The question &quot;How drunk is drunk?&quot; is legally more flexible than &quot;How dead is dead&quot;</strong></p>\n<p>Ultimately, the range of &quot;Too drunk&quot; could be heavily varied - depending on the [...]",
      "score": 6
    },
    {
      "answer_id": 94710,
      "body": "<p>Although some of the answers make a good comparison between retributive and preventative punishment, there is a more utilitarian purpose for this difference.</p>\n<p>Simply put, the law exists so people do not need to  [...]",
      "score": 3
    },
    "..."
  ],
  "question_id": 94665,
  "license": "CC BY-SA 4.0",
  "tags": [
    "criminal-law",
    "driving",
    "sentencing"
  ],
  "score": 23,
  "link": "https://law.stackexchange.com/questions/94665/why-is-drunk-driving-causing-accident-punished-so-much-worse-than-just-drunk-dri"
}
```

## Where it came from

Collected by Yasmin Moslem through the StackExchange API; the notebook `StackExchange.ipynb` in the repository is the collection code [1][8].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] ymoslem/Law-StackExchange dataset card (README). https://huggingface.co/datasets/ymoslem/Law-StackExchange/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=ymoslem%2FLaw-StackExchange - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] Hugging Face Hub API record for ymoslem/Law-StackExchange. https://huggingface.co/api/datasets/ymoslem/Law-StackExchange?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=ymoslem%2FLaw-StackExchange&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ymoslem%2FLaw-StackExchange - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:ymoslem/Law-StackExchange&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] This skill's duplication measurement on the full `train` split, `references/contamination.md` (duplication table) - `question_id` uniqueness. Run 2026-09-23.

[8] Repository file tree for ymoslem/Law-StackExchange. https://huggingface.co/api/datasets/ymoslem/Law-StackExchange/tree/main?recursive=true. Fetched 2026-09-23.

[9] jonathanli/law-stack-exchange dataset card and datasets-server size. https://huggingface.co/datasets/jonathanli/law-stack-exchange/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=jonathanli%2Flaw-stack-exchange. Fetched 2026-09-23.

[10] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=Law-StackExchange&sort=downloads. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as human-written legal Q&A for SFT or score-based preference pairs, after HTML cleaning. CC BY-SA 4.0 share-alike applies to the data.

### The screening row

The row's own note: "all Law Stack Exchange Q&A to Aug 2023; HTML bodies; CC BY-SA." The row carries no flag.
