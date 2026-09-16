# 10th Place Solution

Competition: uspto-explainable-ai
Rank: #10
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522208

First of all, congratulations to everyone who won, got a prize, medals, etc...!
Though we could not become aware of magics like "AND" related tokens as 1 token, we would like to share a brief solution.

### 1. Creating (cpc, ti, ab, detd) Pair Related to Target Patents, and Not Other Patents in Test
First, we calculated all cpc_codes, titles, etc., related to each test rows, and generated pairs. If each token pair was not related to other test patents and not more related to other patents not included in test.csv than patents in test rows, we used this pair.

### 2. Sorting Pairs To Save Token Counts
Imagine a query like below(token counts: 14)
```
(cpc:aaa cpc:bbb) OR (cpc:aaa cpc:ccc) OR (cpc:aaa cpc:ddd) OR (cpc:bbb cpc:ddd) OR (cpc:bbb cpc:eee)
```
By merging pairs which have same token, token counts were saved (14 -> 11)
```
(cpc:aaa (cpc:bbb OR cpc:ccc OR cpc:ddd)) OR (cpc:bbb (cpc:ddd OR cpc:eee))
```
### 3. Final Thought
I was struggling with a reliable validation set like below discussion.
https://www.kaggle.com/competitions/uspto-explainable-ai/discussion/518461
By giving up creating another validation (or test) whoosh index and calculating all patent metadata, everything went fine.

###4. Code Link
Notebook Link(LB: 0.83): https://www.kaggle.com/code/shigeria/uspto-final-sub?scriptVersionId=189656420
