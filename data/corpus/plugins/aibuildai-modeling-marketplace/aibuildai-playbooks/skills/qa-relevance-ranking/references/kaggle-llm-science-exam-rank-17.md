# 17th Place Solution

Competition: kaggle-llm-science-exam
Rank: #17
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446242

Hi, it's just harsh near-missing my first (solo) gold with a big score-gap but it was an intense competition! I don't know if I'll have the courage to try so hard for a solo gold again. 😅 I had shared a summary of my solution yesterday. Now, I've tried to explain it in as much detail and clarity as possible.

---

When dealing with Wikipedia dump data, you realize that you come across a large number of articles. Processing all of them and getting them into a suitable format for a competition is a quite intense process. I also generated all the articles by processing the common dump files we all know. However, I noticed issues with the mathematical formulas and certain Wikipedia LUA templates in publicly available open-source dump extractor outputs.

### STEM Filtering

So, I decided to design a separate pipeline to extract the articles related to STEM in their most original form. First and foremost, I needed to locate the articles related to STEM.
- I downloaded category dumps from wikipedia and **reconstructed the category-links graph** from the sql dumps.
- Ran breadth-first search on STEM main categories for 4 levels and got ~1.1M STEM articles total from all that category tree.

---

### Data
In my final solution, I used three different original data sources.
- **Wikipedia JSON Dump (F1)**: It includes all articles but open-source extractors have LUA template compilation errors at extraction as I said above, which causes parsed texts with no numbers or some garbage in it
- **Local MediaWiki Server TextExtract API (F2)**: Using the filtering technique I described above, I collected 1.1 million STEM article titles. I sent API requests to the MediaWiki server I set up on my own computer to ensure that these articles were parsed in formats closest to the their originals. This way, LaTeX formulas and all LUA templates were parsed without any issues.
- **Cohere WikiPedia corpus (F3)**

I passed all the data I listed above through a Unicode character decode process. Then, I created corpus chunks by shifting 3-sentence windows with a stride of one-sentence.

---

### Retrieval
I noticed that different retrieval methods improved predictive performance. Therefore, I preferred to use three main approaches:
- **FAISS (BAAI/bge-small-en)**: I processed the **F1** data using sentence-embedding models, resulting in an embedding data of approximately ~150GB. I used these without compression during the search phase. I noticed that my retrieval performance decreased when I compressed them.
- **TFIDF**: Since the competition's question-answer texts are paraphrased versions of the original Wikipedia texts, TFIDF was quite effective in capturing exact word clusters. I used TFIDF for ngrams 1,2 with no stop-words on **F2** and **F3**.
- **FAISS + TFIDF**: I also reordered F1 embedding retrieval results with TFIDF (called **F4**)

---

### Ensemble
As I mentioned above, I created four different sources of context. I just ran a single debertav3-large model on all contexts and did a max-ensemble on probabilities. Since I used a single model, I noticed that max-ensemble was quite effective. If it found relevant text in any of the four contexts related to the question, it significantly boosted the probability of the corresponding answer. If it couldn't find the relevant context, the probabilities for all options were similar and relatively low.

---

## Summary
I'm also including a summary diagram of my solution below. It may provide a clearer understanding:


