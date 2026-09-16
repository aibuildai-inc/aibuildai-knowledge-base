# 32nd place solution

Competition: uspto-explainable-ai
Rank: #32
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522214

Thank you to the hosts and participants for hosting this competition! 

## Main Approach
`OR` query consisting of single word(title) and cpc. 

### How to construct an `OR` query
I retrieved candidates of single word(title) and cpc.
　- single word(title): using TF-IDF top-100
　- cpc: all cpc with 50 documents

I checked the relevance by running queries for this candidates.
　- ap@50
　- which of the 50 documents were hit

I used Greedy Algorithm for construct an `OR` query
　- score using which of the 50 documents were hit

(# of words: 24)

## Tried Other Approaches
- title2query with T5: https://www.kaggle.com/code/tinatuna/uspto-t5-based-doc2query-titles (LB: 0.16~)
- using LPProblem: https://www.kaggle.com/code/tinatuna/uspto-using-lpproblem (LB: 0.34~)
- using BeamSearch: (I was unable to resolve Score Submission Error)
- using abstract (LB: 0.43)
