# 18th place solution

Competition: uspto-explainable-ai
Rank: #18
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522380

# Overview of the Approach

Our final approach employed a preprocessing and filtering pipeline designed to minimize overfitting. Initially, we utilized a complex method involving CPC code searches with linear programming, multiple algorithms including decision trees, and logical chains (NOT, OR). Ultimately, we simplified this to a straightforward OR-chain based on single word matches. We created an [SQLite database](https://www.kaggle.com/datasets/raki21/uspto-full-error-corrected) containing filtered information for all publications, implemented in this [notebook using only a single OR-chain](https://www.kaggle.com/code/raki21/simple-or-chain-sqlite-magic-free) to achieve a 0.72+ score (our best submission was only 0.66, this notebook just changed the target score of our approach from precision to mAP, we explain the confusion below).

### Details

0. **SQL Database Creation**:
    - Data was split by words to create an SQL database for inference.
    - Each column was processed using `whoosh.analysis.StandardAnalyzer(stoplist=BRS_STOPWORDS) | NumberFilter()`, followed by removing duplicates to save memory and because single word search was initially prioritized.

1. **Neighbors at Inference**:
    - We loaded the 125k main publications and their neighbors into a dictionary for O(1) retrieval during inference. Additional neighbors-neighbors contributed to a word occurrence dictionary but were not used directly.

2. **Generation of Feature DataFrame**:
    - We identified all words occurring in any neighbor and added them to the feature list. A dataframe was created for each patent with all features as columns and 50 rows indicating occurrence in neighbors, resulting in approximately 10,000 unique words.

3. **OR-Chain Generation**:
    - Features were iteratively selected based on a function penalizing false positives and rewarding true matches. Parameters for penalizing false positives were optimized to select the best scoring OR-chain, focusing on precision rather than mean Average Precision (mAP).

### Problem: Metric Misinterpretation
A significant oversight was the misinterpretation of the scoring metric. Initially, we assumed the metric could be treated as precision since false matches filled up to 50 without considering order. Later, we realized that tf-idf and ordering were used but did not reevaluate our precision-based approach. This led to opting for a 35/15 split instead of a 25/0 split, resulting in lower scores (0.7 vs 0.84).

### Missing Additions
Potential improvements included:
- Faster inference with Ray/GPU to utilize more neighbors.
- Usage of AND for multiple OR-chains.
- A semantically independent NOT-OR chain, leveraging a global word count dictionary to omit highly frequent words missing from all positive cases. This would also need some way to identify omissions that have lots of internal overlap, maybe some semantic similarity measure, as often you would have top 3 omissions being something like [material, materials, metal].

### Final Remarks
We extend our gratitude to the competition organizers, participants who shared their approaches and ideas, and congratulations to the winners!
