# Santa24 10th place - My Takeaways

Competition: santa-2024
Rank: #10
Source: https://www.kaggle.com/c/santa-2024/discussion/560531

# Santa24 10th place - My Takeaways
# Introduction

First of all, I would like to thank the organizers for making this competition possible and my teammates @titericz, @deeper, @lucienderubempre and @nlztrk for their contribution to our team's final result. I would also like to thank the people who shared their ideas, results, experiments and codes publicly. I learned a number of new things from public notebooks, from mathematics to programming.


Below is my personal take on the competition. This is not an exhaustive report, but rather some thoughts that I decided and found time to share. Perhaps my teammates can add something in comments or other posts.


In this competition I have used only Simulated Annealing (SA). The keys to getting good results were:
1. Experimenting with starting sentences for SA and choosing the right ones
2. Good understanding and control of the SA
3. Computational resources

[My public notebook](https://www.kaggle.com/code/egortrushin/santa24-improving-sample-2/notebook), shared in the middle of the competition, is actually pretty close to the code I ended up with. The final, slightly refined version of the code is available at [GitHub](https://github.com/EgorTrushin/Kaggle-Competitions/tree/main/Santa24).

# Details


## Experimenting with starting sentences for SA and choosing the right ones

The choice of the starting sentence for SA was crucial, since in practical optimization SA algorithm was generally unable to reach local minima with a sentence which was too different from the starting sentence, especially for long sentences.

The best example of this is the 5th sample. It was uncovered early on in this competition that the sorted text with stopwords at the beginning **[stopwords] + [alphabetically sorted text]** has low perplexity of about 48 and is a rather good starting sentence for SA. However, it was not possible to achieve perplexity required for gold medal starting from such starting sentence. Instead, it was necessary to reveal that two alphabetically sorted blocks can lead to starting sentences with even lower perplexity and to use it as starting sentence. The idea of two alphabetically sorted text blocks was publicly uncovered in the [public notebook](https://www.kaggle.com/code/woosungyoon/alphabetical-sample-5) by @woosungyoon. After some experimentation with the public notebook, I switched back to my own code and used the following sentence structure as a starting guess:

**[stopwords (without and)] + [alphabetically sorted block 1] + [and and and] + [alphabetically sorted block 2]**

where two alphabetically sorted text blocks were sampled randomly. Starting with such a sentence finally allowed to obtain a solution with a perplexity of 28.5.

## Good understanding and control of the SA

First, I would like to mention the online mini-book, [Simulated Annealing Afternoon](https://algorithmafternoon.com/books/simulated_annealing/), which provides a very good introduction to SA.

#### Generation of neighborhood

The two main operations for generating neighbors are

1. Swap of two words: 1**2**3**4**5 ⟶ 1**4**3**2**5
2. Extracting a word and inserting it in a new position: 1**2**345 ⟶ 134**2**5

At the end of the competition I only used these two. At some point, I found out that it can be useful to have the same two operations only for stopwords. This allows to speed up the optimization for sentences with many stopwords in the initial phase, but is not very useful when the stopwords are already close to some optimal positions. Another operation that I have been using for a while is the cyclic rotation, i.e. 12345 ⟶ 45123, which might help in finding the solution to sample #2.

I have not seen any benefit in using the more complicated ways of generating neighbors that have been publicly suggested. First, I did not see improved results when I tried them. Second, they usually can be derived from more basic operations. Finally, they burn computational resources, if they do not provide useful neighbors with some reasonable frequency. That's why, as already mentioned, I ended up using only two basic operations - (1) swap of two words and (2) extracting a word and inserting it in a new position.


#### Understanding SA basics

Initial temperature, final temperature, cooling schedule, number of temperatures, number of iterations per temperature (if used), monitoring of the acceptance rate, etc. All of this allows one to understand what the optimization is actually doing, whether it is efficient or not, and to design the optimization you want.

I used three types of SA runs, which were designed by appropriate choice of hyperparameters:

[drawing]

(a) **High-temperature exploration**. In this case, I started from large temperature, typically with non-optimzed starting sentence and ran optimization for a while to find some perspective local minima. 

(b) **Warmup run**. Attempt to escape the current local minimum by increasing perplexity using high starting temperature.

(c) **Low-temperature exploration**. SA runs at low temperature, trying to find something overlooked in the vicinity of current local minimum.

The combination of these three types of SA runs allowed me to achieve progress in this competition. It is simple yet efficient.

## Computational resources
Computational resources were very important in this competition. If the final answer is known and the path to it is clear, you can get it even with the computational resources generously provided by Kaggle. However, rapid experimentation with new ideas, tuning and understanding the SA algorithm, etc. depended heavily on the available computational resources.


# Concluding remark
Even a "fun" Christmas Kaggle competition can be tough and a source of learning. I congratulate everyone, who was able to improve their skills and expand their knowledge in this Santa 2024 competition.
