# 2nd place solution

Competition: santa-2024
Rank: #2
Source: https://www.kaggle.com/c/santa-2024/discussion/560540

# Santa 2024 - The perplexity permutation puzzle

Special thanks the organizers for putting on the Santa optimization competitions and to my teammates @solverworld and @zaburo.  They will be posting their own solutions.

[Zaburo's part](https://www.kaggle.com/competitions/santa-2024/discussion/560533)

[Solverworld's Part](https://www.kaggle.com/competitions/santa-2024/discussion/560565)

## Overview

Our solutions all boiled down to two steps:

1. Define a local neighborhood of permutations around a solution to search for a lower perplexity
2. Determine a method for kicking out of a local minimum if there are no better directions to explore.

Problem ID 0 was found with brute force search and publicly known, so I will not discuss it further.

## Local neighborhood for search

This varied from problem to problem, but there were three neighborhoods that I worked with:

(1) Deleting a word and inserting it at another location.  This is a restricted 3-opt and had the smallest effect we found on perplexity since it preserved most of the relative positions of words.

> reindeer mistletoe elf <span style="color:red">*scrooge*</span> <span style="color:green">gingerbread family advent</span> chimney fireplace ornament 

to 

> reindeer mistletoe elf <span style="color:green">gingerbread family advent</span> <span style="color:red">*scrooge*</span> chimney fireplace ornament 

(2) Deleting a phrase (multiple words next to each other) and inserting it at another location.  This covers all 3-opts except those with reversing a segment.

> reindeer mistletoe elf <span style="color:red">*advent scrooge*</span> <span style="color:green">gingerbread family</span> chimney fireplace ornament 

to 

> reindeer mistletoe elf <span style="color:green">gingerbread family</span> <span style="color:red">*advent scrooge*</span> chimney fireplace ornament 


(3) Swapping two phrases.  This would be a double-bridge kick in the TSP literature.

> reindeer <span style="color:red">*advent scrooge*</span> mistletoe elf <span style="color:green">*gingerbread family*</span> chimney fireplace ornament 

to 

> reindeer <span style="color:green">*gingerbread family*</span> mistletoe elf  <span style="color:red">*advent scrooge*</span> chimney fireplace ornament 

We did not include the reversals of one of the segments since would do large changes to the perplexity.  

For problems 1, 2, and 3, running on an RTX A6000 we were able to use neighborhood 3 for a local search space.  For problem 4 we mostly worked with neighborhood 2 but occasionally neighborhood 3, and for problem 5 mostly worked with neighborhood 1 and occasionally neighborhood 2.  


## Kick operations

If we have searched for a while had and found a local minimum, we would then scramble some part of the solution and begin searching again.  For these, @zaburo had the best kick operations, however for solutions to problems 1, 2, and 4 it was sufficient to do either apply a small scrambling of a subset of words, move some words from the front of the solution to the back, or do a few permutation operations using our local neighborhood and try again.  


## Full algorithm

This is the algorithm that I used, @solverworld had a more methodical algorithm.

1. Given a starting solution, search the locally defined neighborhood for a better solution.
    1. If found, set new solution, add the local search neighborhood and scores to a history list, and repeat
    2. If not found, add solution to set of local minima and jump back a random number of steps in the search history and pick a good solution not in the local minima set, and repeat.
2. If no solution has been found after a time, apply a kick operation.

This has the effect of slowly reversing course up a path to a local minimum and looking for a new direction to descend.  If we have found our current search basin is very large, then we try to apply a kick to find a new search basin.


For an example, see [santabasinclimberv7-final](https://www.kaggle.com/code/danielphalen/santabasinclimberv7-final)

## Details for Problems 3 and 5.

For problems 3 and 5, the minima found were in very large spaces and we were having trouble finding good ways to better search the space, even with kick operations.  So, some extra methods were used to generate starting places.

### Problem 3

For problem 3, @zaburo had the idea to pin the *last* word and generate minima.  This seemed to span the space and allowed us to jump out of the 197.5 minimum to the 191.5 minimum.


### Problem 5

We spent a lot of time with solutions that began from `(stop words) (sorted remaining words)`, but were stuck at 32.5 for a long time.  We then realized that Problem 5 was made up with words from Problems 1, 3, and 4, so started generating starting solutions based on combinations of those problem solutions.  This allowed us to jump out of this minimum.  Finally, @zaburo implemented a custom kick based on our local minimum that put us to the 28.5 solution.


## What did not work for us

1. Simulated Annealing or its robust cousin, Late acceptance hill climbing.  It seems they were not methodical enough in terms of climbing out of basins and sampling larger amounts of the search space.
2. Attempts to break the problem down into phrases and do a brute force search over those phrases, or to fix some phrases and apply our TSP-like methods.
3. Branch and Bound.  It was not good enough at eliminating part of the search space, so did not allow the search to converge appropriately.
4. A Multi-Arm Bandit type solution using random permutations.
