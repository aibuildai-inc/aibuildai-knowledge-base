# 22nd Place Solution - Simple TF-IDF

Competition: uspto-explainable-ai
Rank: #22
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522327

Thank you for organizing this very interesting competition. And thank you to all participants.
The 22nd place solution is a simple solution using only “title” and “cpc”.

# Creating candidates

### Step 1: 90 candidates by TF-IDF (title)
- stopwords='english'
- ngram=(1, 3)
- max_df=100
- min_df=3
- If a word is in another word set, exclude it
- Exclude words with less than 5 letters

Step 1 may not reach 60 because the title is too short. In that case, add Step 2 to the candidates.

### Step 2: 60 candidates by TF-IDF (cpc)
- max_df=0.01

The sum of candidates in Step 1 and Step 2 may not reach 30 because some publication_numbers have very little information on title and cpc. In that case, add Step 3 to the candidates.

### Step 3: 10 candidates by word occurrence of title
- Split title into words and sorted by number of occurrences (max 10)

# Annealing algorithm

Using these candidates, run the Annealing algorithm. However, I am unable to understand the annealing algorithm and use it directly from the following two excellent public notebooks.
@tubotubo @andrey67 I really appreciate it.

https://www.kaggle.com/code/tubotubo/uspto-simulated-annealing-baseline
https://www.kaggle.com/code/andrey67/uspto-annealing-lb-0-31

Of course, I could not find MAGIC at all. It is very amazing who found it!
