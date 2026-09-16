# 3rd Place Solution

Competition: uspto-explainable-ai
Rank: #3
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522639

I would like to express my sincere gratitude to the organizers for creating this fascinating optimization challenge, and to the dedicated competitors who worked tirelessly to fix Whoosh, ensuring a valid competition.

# Query Optimization
As many participants have pointed out, tokens can be AND-concatenated using the following method without increasing the "number of tokens" as measured by whoosh_utils.count_query_tokens:
`ti:token1-token2`

The final query consists of these subqueries OR-concatenated, such as:
`(ti:token1-token2) OR (ab:token3-token4-token5) OR …`

Regrettably, I was unable to discover the stronger magic that incorporate tokens from different fields into a single subquery.

# Subquery Search
For each sample, up to several thousand subquery candidates were generated. For every small subset of patents (typically size(subset) ≤ 3), tokens common to all patents in the subset were selected and adopted.

# Subquery Selection
Mixed Integer Programming (MIP) solvers were employed to determine the optimal query. In this context, "optimal" refers to maximizing the number of target patents found using a random test_index that covers the same number of patents as the train_index. More details can be found in the shared notebook.

# Key Strategies
1. Implementation of subquery search in C++ with multithreading and aggressive algorithm optimization for improved search speed.
2. Consideration of all tokens found in Title, Claim, and CPC fields, while using only the 100,000 least frequent tokens for Abstract and Description fields to reduce computational complexity.

# Areas for Improvement
1. Utilizing the stronger magic.
2. Implementation with CUDA for more intensive search.

[best submission with neater code](https://www.kaggle.com/code/cnumber/uspto-bestsub)
[codes for preprocessing](https://www.kaggle.com/datasets/cnumber/uspto-preprocess)
