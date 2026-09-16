# 5th Solution - CPMP part

Competition: santa-2024
Rank: #5
Source: https://www.kaggle.com/c/santa-2024/discussion/560597

First of all, let me thank @horeahoreastefan . I am not sure I would have found the best score on all samples without him. When we teamed he had the best score for sample 3, and my method doesn't seem to be able to find it without some of his input (some word grouping). My method works great for the other samples: it finds the best score more than half the time when starting from a random shuffle of the original text words.

I will focus on describing my local search and few of the things I tried that did not work. I am writing this before reading other teams write up, hence there maybe be some redundancy.

# Simulated Annealing Variant

I use a variant of simulated annealing (SA). In SA, one works with one solution to the problem and modifies it iteratively. If the modified solution score is better than before, then it is kept. If the modified solution score is worse than before, then it is kept if it passes some test. 

Most people use an exponentiation of the old score minus the new score to decide if the new solution is kept. There is a more efficient way, and as effective, to do this: the new solution is kept if its score is lower tan some global upper bound. That upper bound is lowered from time to time.

As in the exponentiation based SA, the best score is maintained throughout the search. 

# Multi Point Search

In order to use GPU as effectively as possible I run N SA in parallel, where N is the largest batch size for evaluating N texts with the competition metric. For instance, I used a batch size of 104 for sample 5 when using a A100 GPU.

The algorithm is similar to SA.

```
Start with S = {N random shuffles of original text} and a margin M
Repeat until stopped by user
    S1 = empty set
    for each solution in S
        Modify S and add to S1
    Compute the score for each text in S1 (in one batch on GPU)
    if one of these score is better than best score, then update best score and best text
    Keep in S1 the solutions with score smaller than best score plus M
    For the other ones, perform  crossover with the best solution, and add it to S1
```

The crossover takes as input two texts and produces a new text from it

The use of crossover is reminiscent of genetic algorithm (GA) but this is the only little bit of GA used. We do not apply crossover to all texts in S, nor do we keep the N best score from S1.

The purpose of the method is to maintain a diversity of optimization paths by running them in parallel as much as possible. GA tends to focus on a subset of the space around the best solution so far.

# Moves

Three kind of solution modifiers are used, all inspired by ATSP methods. We work with sequences (python list) of words, and construct text only when computing scores. The <bos> and <eos> tokens are added when computing the score as well.

1. **k-opt** The word sequence is split into k segments. The segments are shuffled and concatenated to yield the new word sequence. The choice of k is random up to a max value (e.g. 10). We use some variants of this, for instance fixing some of the segments and permuting the others. This generalize the permutation of a subset of the words.

2. **remove - insert** A subset of the word sequence is removed, then removed words are inserted back. One variant uses random insertion. Another one tries every possible insertion location and picks the best one. There are also variants in how the removed words are selected. It could be a random subset of size k, or a sub sequence of length k. The latter seems more effective. As before, k is selected randomly up to some maximum value

3. **double root and stem**. This is from a [paper on ATSP heuristics](https://leeds-faculty.colorado.edu/glover/469%20-Traveling%20Salesman%20-%20Doubly%20Rooted%20Ejection%20Chain%20ATSP_Final.pdf)

Explaining the method here would be too long, better read the paper. Let me just say that instead of working with a word sequence, a more complex graph structure is used, with either two cycles and a path between them, or 3 paths with same end points. That structure is modified with specific moves, then translated back into a sequence. In order to give a taste of it here is how the structure is transformed into a sequence:



# Speeding up things

The bottleneck is in GPU usage. Our code uses GPU quite effectively, with 98% average use. The main way to speed up things was to use a global cache of text scores. It is a python dictionary whose keys are texts, and values text scores. 

We did not optimize much the solution moves implementation as time spent there was not relevant. We preferred to ensure easy modification of the code.

# Discounted perplexity

After a while it became clear that modifying a word sequence near the start of it had a much higher influence on the score compared to modifications near the end. As a result, local search quickly focus on the end of sequences and left the beginning fixed. 

@horeahoreastefan had the idea to restrict moves to some prefix, which forces the local search to explore moves near the start. This moved us closer to best score on sample 5 during the competition. 

I explored another way. Besides computing the score using the competition metric, I maintain for each token position the average of logit value. Then, for each text,  I compute a discounted score by first dividing its logits by the average logit value at that position, then take the mean and exponent. That way, score modifications  amplitude are normalized across all positions. The discounted score is used to decide if a modified text should be kept or not. That way, modifications near the start of the sequence become as probably as modifications near the end.

This concludes the description of the method I used.

# What did not work

I tried ideas similar to what @simjeg shared. As in Simon's method, I use a permutation matrix as the first layer of the model. I used the logits of next tokens at each position restricted to tokens appearing in the text. This gives me a square matrix. I then apply Sinkhorn to it.  Then KL divergence between the Sinkhorn logit matrix and the input permutation metric is computed, and back propagated to update the permutation matrix. 

This worked somehow, but nowhere near the local search results.

In another variant I compute an ATSP solution using the logit matrix as cost. Then. the resulting solution is used as the next text input. This is iterated

It didn't work at all.

I then used an exponential moving average of the next token logits before Sinkhorn, with the hope that the values would converge to something that led to optimal ATSP. It worked better, but not as well as the local search described above.

A third idea I explored was to train a model to predict good sequences. I used Monte Carlo tree search (MCTS) with RL. It did not work either. Maybe I should have insisted, but local search worked well enough.
