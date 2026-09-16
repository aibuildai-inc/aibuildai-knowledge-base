# 28th Place Solution

Competition: uspto-explainable-ai
Rank: #28
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522415

Thank you to USPTO for organising and hosting this competition, and a big congratulations to all the winning teams.

## Solution Overview

The overall approach was to construct a query in the form: “rare_cpcs_docs OR ((sub-query) AND (sub-query) AND …)”. We made use of a global frequency index for CPCs and an IDF lookup table for Abstract & Claims, which is discussed in detail below.

- We discovered early on that just using CPC codes was very powerful for retrieval. To reduce our search space, we constructed a ‘rare_cpcs_docs’ query, which is an OR-wise concatenation of CPCs. This was only for documents that had very few global hits on CPCs (< 50). Testing locally, this was always around 5-10 documents.
- The aim for each ‘subquery’ is to obtain common concepts to cover the remaining ~40 documents. We did this through a combination of using CPC and keywords extracted from the Abstract & Claims using KeyBERT.

We focused on rare CPC & rare keywords that shared mostly within 50 docs (i.e CPC with high local hit and low global count, keywords with high local hit and high global IDF), then we greedily picked the best performing entity until all documents were covered (Greedy Set Cover).

- This process was repeated to construct as many iterations of a ‘sub-query’ as possible until we ran over the 50 token limit (no ‘magic’ token limit)

**Generating ‘sub-query’**

We performed a lot of experimentation in generating these sub-queries, but none led to huge gains:

- Using the Title field
- Filtering out overlapping words across sub-queries, e.g., if “electrode” is used in sub-query-1, filter out “negative electrode” so it’s not used in subsequent queries
- ADJ/NEAR within ngram (>=3gram) gives more choices of rare keywords that have higher hit within 50 documents. However, it costs more query term than other operators.

**Constructing Global Indexes**

We analysed the entire set of parquet files (10+ million individual patents) to construct these indexes:

- Frequency count of CPC codes
- Unigram & bigram IDF values for Abstract
- Unigram & bigram IDF values for Claims

The aim was to use these lookup tables to penalise high frequency terms, thus reducing the number of extraneous documents we retrieved.

### Other Ideas

- Our first entry consisted entirely of CPC codes in the form of “cpc:(a OR b OR c OR …)”. The codes were chosen based on the factor of local frequency, i.e., we prioritised codes that covered most of the 50 documents first; a simple greedy set cover algorithm. This simple query scored 0.37 on the leaderboard.
- We also tried to focus the ‘subquery’ on each part of a patent such as (ti:subquery) (clm:subquery)(cpc:subquery). This would help to detect the common patent landscape of each part over 50 docs. Due to the small query size on each subquery, this solution didn’t produce the expected improvements.

### Tips & Tricks

- We used Polars + multi-threading to very quickly process and store the 125,000 relevant patents in memory for instant retrieval
