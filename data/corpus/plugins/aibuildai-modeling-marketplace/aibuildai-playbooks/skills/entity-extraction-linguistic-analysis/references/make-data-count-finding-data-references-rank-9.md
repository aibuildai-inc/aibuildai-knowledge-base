# 9th place Solo Gold solution

Competition: make-data-count-finding-data-references
Rank: #9
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/9th-place-solution

This competition is my first solo gold (and first solo medal overall 🎉)

# **Brief Summary**

**Accessions**
- MDC corpus v4 (eupmc only)
- Extracted sentence containing the ACC_ID and calculated the sentence similarity using Qwen3-Embedding-0.6B w.r.t. ```seed_sentence = "DNA Deposition\nThe following information was supplied regarding the deposition of DNA sequences:\nThey are available at\nGenBank: PRJNA664798. BioSample: SAMN16233641, SAMN16233642, SAMN16233643, SAMN16233644, SAMN16233645."``` - if >= 0.667 and contains submit/deposit then Primary else Secondary

Postprocessing
- Due to the training set having articles 74/89/111 FPs for ENA/Proteins, I hedged by limiting the number of such predictions per article to 64 (in case there may be TPs too).
- Any ACC_ID that falls within a range in the PDF text are deleted (e.g. KT123456-KT123489 deletes up to 34 ACC ID if predicted)


**DOI**
- MDC corpus v3
- Step 1: extracted DOI links using Qwen2.5-7B with surrounding context
- Step 2: extract title, author, published year from beginning of paper using Qwen2.5-32B
- Step 3: classify Data vs Literature using Qwen2.5-32B 
- Step 4: classify Primary vs Secondary using Qwen2.5-32B. Set all DRYAD to ```Primary```
- Step 5: Only for DRYAD IDs, append predictions which are present in the corpus but not in the article and assign them as ```Primary```

# **Code**

GitHub repo: https://github.com/bogoconic1/9th-place-kaggle-mdc-finding-data-references
Notebook: https://www.kaggle.com/code/yeoyunsianggeremie/9-mdc-final-solution-notebook

# **Design Choices**

- I decided to not use a probabilistic solution for Accession IDs, since after manually analyzing most of the PDFs, I could not find any real patterns that I believe a classifier would do well in. The data was extremely noisy. Just using a simple rule ```let S = sentence containing the ACC ID. If deposit in S or submit in S, then Primary else Secondary``` captures 95% of the Primary accessions., Subsequently, I used embedding similarity to filter out those false positive sentences which may happen to contain the keyword but are irrelevant.
- For DOIs, I used Qwen2.5-32B-Instruct. Didn't make any model changes other than testing 72B which was way too slow for experimenting and did not provide much improvements. I had the idea of generating synthetic data to fine-tune LLMs for classification, but my CV results showed that all of GPT-5, Claude 4 and Grok 3 performed similarly to Qwen2.5-32B-Instruct so I abandoned the idea and just focused on zero-shot with proper context engineering.

# **Final Selected Submissions**

(1) Public: 0.853, Private: 0.729 (all BioSample/BioModels as Primary)
(2) Public: 0.797, Private: 0.745 (without above rule)

Best unselected: Public 0.802, Private: 0.750

# **Failed Ideas**
- Using publication date vs dataset creation date to classify Primary/Secondary - it did not improve CV
- Sweeping through the article in chunks of 2048 tokens to identify accessions/DOIs with Qwen 32B (similar to NER). It did not improve both CV and LB.
- I have no idea why, but Accession IDs and DOIs have just 1 degree of freedom in the training data. I tried dropping Accessions for articles with a predicted DOI, but the score decreased. 
- Reasoning for classification
- Using a different PDF parser (e.g. Marker)
- Semantic/Keyword search for context engineering


# **Footnote**

I was stuck at 0.64 public score for one week. It's discouraging, but I knew to get a decent result I need to perform deep debugging on why my score fails rather than just giving up and falling back on publicly available ideas. I stopped looking at the leaderboard and shoved any thoughts that I wasn't improving due to the lack of progress over the past year to the back of my head, and just focused on debugging it until I succeed. Eventually, I found the root cause and improved to 0.82 public scores within five days. I won't be successful without struggling, as per [Principles by Ray Dalio](https://www.principles.com)
