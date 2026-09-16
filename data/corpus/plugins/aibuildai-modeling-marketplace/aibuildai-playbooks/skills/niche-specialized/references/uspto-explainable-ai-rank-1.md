# 1st place solution

Competition: uspto-explainable-ai
Rank: #1
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522233

Thanks to the hosts for this interesting competition. Congratulations to the winning teams. Thanks also to [@tomyanabe](https://www.kaggle.com/tomyanabe) for competing with me.

# summary

* Simulated Annealing
* Only AND, OR
* Omission of AND using -

# query

The format of the query is as follows:
(ti:word1-ti:word2-detd:word3-...-cpc:wordN) OR ...

* The query is composed only of AND, OR
* By connecting words with -, the AND token can be omitted
* cpc cannot be omitted with -, so cpc is placed last
* Use all words for cpc, title, abstract
* Delete words with high frequency (100,000 or more) for claim, description

# candidate generation
Define a sequence of words connected by AND as a subquery.
Generate candidates for subqueries to be used in the query through the following steps:

1. Generate a set of words to be used in subqueries
    * All words possessed by a single target
    * Common set of words possessed by two targets
2. Sort words by the number of elements
3. Add words until the patent set consists only of targets, and make it a subquery
    * The patent set is obtained by taking the common set of each word
    * If there are non-targets after combining all the words, that subquery is not a candidate

## tips for improvement
* Reduce computational complexity by adding words in ascending order of elements
    * The complexity of calculating the common set of two sets s, t is min(len(s), len(t))
* Speed up the calculation of the common set using cupy
    * 2-3 times faster compared to using set(a) & set(b)
    * cp.intersect1d(array1, array2)
    * On the final day, speeding up with this allowed using all cpc, title, abstract, and the rank improved from 3rd to 1st
* Reduce memory usage by placing only patents appearing in test.csv (2500*50) and the words those patents possess in memory

# Simulated Annealing
* Combine subqueries with OR
* Neighborhood
    * 50% chance to add one unused subquery
    * 50% chance to remove one used subquery
* Score function
    * The number of targets included in the search results of the query
    * Consider only the number of targets as candidates are subqueries with zero non-targets
* Duplicate removal
    * Reduce the number of candidates by removing duplicates, as subqueries with the same target set do not need multiple candidates

# code
https://www.kaggle.com/code/tanakar/sa-cpc-title-abst-clm-desc-10-5-allpub-cupy-sub
