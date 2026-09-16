# 47th place solution

Competition: uspto-explainable-ai
Rank: #47
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522205

Thank you to the hosts and participants for hosting this competition. I learned a lot.

My solution is as follows.

### Query check for single word and CPC

I checked the relevance by running queries for the words and CPC obtained by tfidf. This increased the score by about 0.04.

```python
    for j, word in enumerate(topk_words):
        ti_query = f"ti:" + word
        cand = whoosh_utils.execute_query(ti_query, qp, searcher)
        ti_score = ap50_true(cand, target)
        ti_score += (len(topk_words) - j) * 0.00001
        ti_scores.append(ti_score)

    for j, cpc in enumerate(topk_cpc):
        cpc_query = f"cpc:" + cpc
        cand = whoosh_utils.execute_query(cpc_query, qp, searcher)
        cpc_score = ap50_true(cand, target)
        cpc_score += (len(topk_cpc) - j) * 0.00001
        cpc_scores.append(cpc_score)
```

### Difficulty assessment

When meta_i was divided into 5 parts and the CPCs obtained by tfidf were all the same, the score tended to drop significantly.

Perhaps there was an adjacent patent that did not have a CPC.

When that condition was met, a two-word search was added to give more importance to the word.

```python
    meta_i_list = []
    for j in range(5):
        start_index = j*10
        end_index = min(start_index + 10, len(meta_i))
        if start_index >= len(meta_i):
            break
    meta_i_list.append(meta_i[start_index:end_index])

    cpc_mat_list_d = [cpc_cv_tfidf.transform(m.get_column("cpc")) for m in meta_i_list]
    cpc_idx_list_d = []
    for cpc_mat_d in cpc_mat_list_d:
        X_cpc_d, cpc_idx_d = select_top_k_columns(cpc_mat_d, k=4)
        cpc_idx_list_d.append(cpc_idx_d)
    cpc_idx_list_d = np.unique(cpc_idx_list_d)
    print(len(cpc_idx_list_d))
    difficulty = False
    if len(cpc_idx_list_d) <= 4:
        difficulty = True

    if difficulty:
        X_ti, idx = select_top_k_columns(ti_mat, k=100)
        X_cpc, cpc_idx = select_top_k_columns(cpc_mat, k=30)
    else:
        X_ti, idx = select_top_k_columns(ti_mat, k=30)
        X_cpc, cpc_idx = select_top_k_columns(cpc_mat, k=50)
```

### Random Judgment

The area that had to be explored was so large that random judgment would have helped improve the score.

```python
    def move_random(self):
        p = 0.65 + 0.05 * np.random.choice(range(6))
        self.use = np.random.binomial(1, p, len(self.words))
        while len(self.words) >= 1 and np.count_nonzero(self.use == 1) == 0:        
            self.use = np.random.binomial(1, p, len(self.words))
        
        return self
```
