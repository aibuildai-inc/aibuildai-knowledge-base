# 11th Place Solution

Competition: uspto-explainable-ai
Rank: #11
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522207

Thanks to the organizers for hosting the competition and thanks to everyone who contributed code and ideas. 

## Overview
- Queries consisted of infrequent tokens connected by `OR`
- Used single words and `ADJ` pairs
- No magic, and simple rules to find decent token combinations for query

## Validation
Used 2000 patents and the 100 nearest neighbors. I pulled the embedding data using BigQuery and then created dataset with 100 nearest neighbors. This just about perfectly matched the leaderboard so I wonder if the test set also used more then the 50 nearest neighbors.

## Approach
Early on I realized that we can identify individual patents by very infrequent words (plenty of misspellings). Just using infrequent tokens and and `OR` gets 0.77. I created datasets containing the frequencies of all the words for all patents, and then datasets containing only the less frequent tokens. 

In addition to uniquely identifying patents with infrequent tokens I also identified groups of patents that shared the same infrequent token. 

I had to stop competing on Kaggle a couple of weeks into this competition and this is were I left my solution. 

In the last few days seeing that I could maybe get a gold meddle if I improved a little and I added directly `ADJ` pairs (no gap, or distance of 1) for title, abstract and claims. If I could get at least two of the target patents and the adjacent pair appeared less then 8 times in the full dataset then I would use the pair. This brought my score to 0.81. I got lucky on the shake and just squeezed into the gold section (sorry dt). 

My approach was not very optimized and I am excited to see some interesting solutions being posted. I didn't find any of the magic for lengthening queries, but relieved that I didn't miss anything super obvious to improve score to upper 90s.
