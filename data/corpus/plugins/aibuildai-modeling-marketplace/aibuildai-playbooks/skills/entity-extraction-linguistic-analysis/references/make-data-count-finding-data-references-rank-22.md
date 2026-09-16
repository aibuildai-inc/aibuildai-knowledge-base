# 22nd Place Solution | ML

Competition: make-data-count-finding-data-references
Rank: #22
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/23-place-solution-ml

Thanks to the host and participants. It was a rather tiring and complicated competition.

Our solution was as follows:

## DOIs Pipeline

1. Process all PDFs to extract the body section and use a special ML-based approach to extract a clean Header (to get authors’ names) and Reference sections.
2. DOI Extraction:

* Fixed-length patterns (safe to use regex):

```python
pangaea_pattern = r"(?:1\s*0\s*\.\s*1\s*5\s*9\s*4\s*/\s*)?p\s*a\s*n\s*g\s*a\s*e\s*a\s*\.\s*(?:\s*\d\s*){6}"
pasta_pattern = r"(?:1\s*0\s*\.\s*6\s*0\s*7\s*3\s*/\s*)?p\s*a\s*s\s*t\s*a\s*/\s*(?:\s*[a-f0-9]\s*){32}"
doi_pattern = f"(?:{pangaea_pattern})|(?:{pasta_pattern})"
```

* Variable-length patterns (not safe to use regex, needs LLM):

  * We first use regex to highlight possible DOIs in the article (based on this pattern: `r"1\s*0\s*\.\s*\d{4,9}\s*/"`).
  * Then we create a right window with 80 chars and ask Qwen 7B or 14B to extract it.
  * For special brand patterns (without the standard DOI prefix 10.xxxx/), we handle them separately with special processing and prompts.
  * The CV result of this stage was strong (we mainly care about recall):
    `~doi - f1: 0.7597 [tp: 294 / fp: 157 / fn: 5]` → so we missed only 5 DOIs.

3. Filtering Non-data DOIs:

* To reduce FPs, we filter out non-data DOIs using fine-tuned Qwen 14B.
  `~doi - f1: 0.8985 [292 / 59 / 7]`

4. DOI Type Classification:

* We use fine-tuned Qwen 14B with 4 prompts depending on the available information (A clean Author list available or not, a clean Ref available or not ..,etc.).
  `~doi - f1: 0.8554 [278 / 73 / 21]`

## Accession IDs Extraction

We use regex and the public solution for classification.

---

We submitted two solutions to the final evaluation:

1. Pure ML (Main solution):

   * Public LB: 0.80
   * Private LB: 0.613

2. ML + MDC corpus:

   * Public LB: 0.846
   * Private LB: 0.647

---

**Note:**
On the last day we discovered that other teams used the MDC corpus to filter out FPs. So we submitted another run with the same approach to stay on the same page. However, we feel this isn’t a clean solution @inversion , because using it raises the following question:
*"How about inference mode when the host wants to use your code?"*
He will get nothing, because you treat all new DOIs as FPs (non-data).

---
