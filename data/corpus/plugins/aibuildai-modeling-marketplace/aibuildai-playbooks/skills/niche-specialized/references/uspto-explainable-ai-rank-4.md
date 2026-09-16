# 4th Place Solution

Competition: uspto-explainable-ai
Rank: #4
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522200

First of all, congratulations to the winning teams.

It is very unfortunate that there were some magics regarding query construction. I only became aware of them 10 days ago and have since spent a lot of time considering how to use them. 

I will explain the solution using magic (LB 0.98) and the solution not using magic (LB 0.91).

## Solution with magic (LB 0.98)

You can save AND tokens by constructing the query as follows. The number of tokens in the example below is “1”.

> ti:”token1”detd:”token2””token3”ti:”token4”’

This is equivalent to:

> ti:token1 AND detd:token2 AND token3 AND ti:token4

In my solution, I divided the target patents into 25 pairs and connected up to 25 common tokens by AND tokens in order of decreasing number of occurrences. The final query has the following form:

> (”token1”ti:”token2”…detd:”token25”) OR (”token26”ti:”token27”…detd:”token50”) OR …

To determine which pairs to match, I used the product of the token occurrence probabilities. After matching, each partial query is run against the test-index, which contains all patents in test.csv, and if other than the two targets are hit, they are not used for the final query. Next, hit each unused target one by one with the extra tokens. (As in the case of pairs, I only greedily combine tokens that occur infrequently.) Finally, I added common tokens whenever possible to ensure that the length of the query did not exceed the upper limit of 10,000.

## Solution without magic (LB 0.91)

Simulated annealing for queries in the following format:

> cpc:CPC1(token1 OR token2 OR …) OR cpc:CPC2(…) … OR (token1 OR … OR tokenN)
> 

The evaluation function in SA is the expected value of mAP. This is not an exact expectation, but it works fast enough and accurately enough. Please check `state.py` to be released later for details. ([here](https://github.com/penguin-prg/uspto-4th-place-solution/blob/main/src/solver/state.py#L189))

Pre-processing of this solution requires about several weeks in my environment. Various speedup and disk space-related tips were made, for examples:

- Pre-file tokenized text data
- Compress the file as .bz2
- Use a Key Value Store called leveldb
    - Directly handling .bz2 is faster and easier to parallelize, but uploading to the Kaggle Dataset is very cumbersome when the number of files is too large.
- When using leveldb, the data itself is consolidated into a single text file, and the DB stores the range to be read
    - i.e, key → (leveldb) → range → (.txt.bz2) → data
    - Storing the data directly in the DB has deteriorated the DB construction and access speed
- Use hash to split the DB
    - When uploading to kaggle notebook, if the DB is larger than 20GB, split it into several DBs.
    - In this case, a hash function is used to determine which DB a key belongs to. This eliminates the need for additional dict or DB.
- For large pre-processing, save a flag to avoid repeating the completed process
    - Processes may be interrupted by errors due to OOM or unexpected corner cases.
    - When resumed, processes that are flagged as completed will be skipped.
- etc…

# Conclusion

Again, it is very sad that such magics were uncovered at the end of the competition, as this was a very interesting task. I found several other magics in addition to this one. (e.g., a way to express an exact match of multiple tokens in a single token) As far as I know, there is no precedent of LB being recalculated after deadline in past competitions due to leaks or other defects, so LB will probably be finalized as is. Fortunately, however, as far as I can see from LB, most of the top participants did not use these magics until at least 2 weeks before the competition deadline. I look forward to reading their solutions without magics.

# Code

- final submission (0.98) : https://www.kaggle.com/code/ryotayoshinobu/uspto-4th-place-solution-w-magic-lb0-98
- final submission (0.91) : https://www.kaggle.com/code/ryotayoshinobu/uspto-4th-place-solution-w-o-magic-lb0-91
- github: https://github.com/penguin-prg/uspto-4th-place-solution
