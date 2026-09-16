# 3rd place solution

Competition: santa-2024
Rank: #3
Source: https://www.kaggle.com/c/santa-2024/discussion/560620

The write up is completed by @asalhi  and I make the post because he had problem posting it. 

# **“Sorted stopwords, Sorted verbs, and Sorted Other words” are all you need!**

First of all, we would like to thank Kaggle for this great competition, we enjoyed it a lot, it was fun and informative. 
Also, I would like to deeply thank my teammates, it was a wonderful experiment 🙏

Regarding our solution, we used Simulated Annealing (SA) with modifications related to candidate generation. 

We will be sharing two approaches (one that worked best with samples 0 to 4 and achieving 30.x in sample 5) and the other that helped us reach 28.5x , both sample codes will also be shared. 

# **Overview of the first approach:**
## **Simulated Annealing with Custom Weighted Candidate Generator:**
Sample code for 1st approach can be found here: https://www.kaggle.com/code/asalhi/3rd-place-sa-weighted-candidate-generator-sol1

**Batched Metric:**
We used the metric with batches to speed up the process of calculating the scores. 

**Custom Candidate Generator:**
We created a custom candidate generator with the following features:

1- We defined different operations (text transformation operations) : 

```
operations = ["swap","reverse","removeinsert","removeinsertn","rotate","shift","swap_adjacent","circular_shift","reverse_all","scramble_except_first","block_shift","block_swap",]
```
2- The selection of which operation to use is not purely random, we created a weighting system to give importance to operations based on the score they produce.

Here are the main methods used for giving weights to candidates : 
```
def compute_word_importance(self, words):
        """
        Compute word importance based on perplexity scores.
        """
        importance_scores = {}
        for word in words:
            # Get perplexity for each word
            perplexity = self.scorer.get_perplexity([word], batch_size=self.batch_size)
            importance_scores[word] = 1 / perplexity[0]  # Higher perplexity -> Lower importance
        self.word_importance = importance_scores
```
```
def update_weights(self, operation_scores):
        """
        Updates the weights for operations based on their performance.
        :param operation_scores: A dictionary mapping operations to their scores.
        """
        scores = operation_scores.values()
        min_score = min(scores)
        max_score = max(scores)

        # Reward-punishment mechanism
        for i, operation in enumerate(self.operations):
            score = operation_scores.get(operation, max_score)
            if score == min_score:
                self.operation_weights[i] *= 1.2  # Strong reward for best operation
            else:
                penalty = 0.9 + (0.1 * (score - min_score) / (max_score - min_score + 1e-6))
                self.operation_weights[i] *= penalty

        # Normalize weights to avoid extreme bias
        total_weight = sum(self.operation_weights)
        if total_weight <= 0 or not math.isfinite(total_weight):
            # Reset weights to equal distribution if normalization fails
            self.operation_weights = [1.0] * len(self.operation_weights)
        else:
            self.operation_weights = [w / total_weight for w in self.operation_weights]
```
**Enhanced Simulated Annealing:**
We used the Simulated Annealing algorithm with the following enhancements : 
1- Tracking operations (text transformation operations score and use that to update the selection of candidates.
2- Taking texts with "equal best score" into consideration, this help escape dead-end paths.   
**Note:** This happens with A100 GPUs at least, where two texts (or more) can have the same score.
3- No improvements: when no improvements happen we reselect a text and start from it (basically some shuffling or back to the best text ), it's based on experiments.

**Initial Text “Starting Point”:**
What played a good role in achieving better scores was the initial text that we used to start with, after different experiments we found that the following worked best : (Sample IDs from 0 to 5)
1- Sorting the text alphabetically: Worked Best with Sample 2
2- Stopwords sorted then alphabetically sort other words: Gave good starting point for Samples 3 and 4
3- Random Starting point (running the code many times in parallel): gave faster convergence and the better result for sample 3
4- **Sorted Stopwords, Sorted verbs, Sorted Other words**: Gave best-starting points for Sample 3 and 4 and the most important sample 5 ( even if the starting score is not as low as all the text sorted)

## **Overview of the second approach:**
Sample code for 2nd approach can be found here : https://www.kaggle.com/code/siavrez/3rd-place-sa-weighted-candidate-generator-sol2

For the second path we started with a simple simulated annealing model with only swapping 2 words and different ideas were added based on different experiments:

1- Adding more operations: *BlockMoveOperation, BlockSwapOperation, ReverseBlockOperation, ShuffleBlockOperation, CyclicShiftOperation, ShiftOneOperation, TripleMoveOperation, QuadMoveOperation, InterleaveOperation, SplitMergeOperation, PivotRotateOperation, WindowSlideOperation, CrossBridgeOperation, RotateBlockOperation, MultiSwapOperation*.
2- Cyclic cooling down.
3- Limiting the range of acceptable energies in the next iteration.
4- Testing different strategies for starting sequences (clustering, sorting, split sorting). I think this one was the most important one for sample 5 and the best combination for us was, **sorted stopwords + sorted verbs + sorted others**.
5- Using the best sequence from one run and feeding it to another variation of the model by randomly selecting only a subset of operations.
6- Using top-k for the selecting next text and choosing one randomly or weighted based on distance from best energy (instead of only considering the best score)
7- Exploring top-k energy parallel paths and selecting the next best text based on that (instead of selecting best explore the path for all top-k).
8- Using operation weighting based on the first approach.
