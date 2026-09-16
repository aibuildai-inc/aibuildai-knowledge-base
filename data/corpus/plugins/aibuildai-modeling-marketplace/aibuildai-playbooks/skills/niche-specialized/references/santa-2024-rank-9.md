# 9th place solution

Competition: santa-2024
Rank: #9
Source: https://www.kaggle.com/c/santa-2024/discussion/560601

I sincerely appreciate the organizers for planning such an interesting competition.

I used **a genetic algorithm using Edge Assembly Crossover (GA-EAX)** to search for solutions to all six test cases.
(For ID=0,1,2, the final solution was reached using SA, which was tested initially)

# Outline of GA procedure



1. Create the first generation using Simulated Annealing (SA)
2. Generate children using Edge Assembly Crossover (EAX)
3. Apply local search and SA to the children to generate individuals with low perplexity (raised children)
4. Select the next generation from the parent individuals and the raised children
5. Repeat 2.-4.

# Initial individuals and neighborhood operation
- Up to ID=3, random initial individuals were used, and random 3-opt was used for neighborhood search.
- For ID=4, individuals with stop words at the beginning of the sentence were used ([stop words] - [non-stop words]), and random 3-opt was used for neighborhood search.
- For ID=5, stop words were placed in the first block, and the other words were randomly divided into two groups, which were then sorted and placed in the second and third blocks. Such sentences were used as the initial individuals ([stop words] - [non-stop words] - [non stop words]). For the neighborhood operations, operations that move words between groups while maintaining alphabetical order were prioritized.
 - In EAX, crossovers that break up groups were allowed.

# Generation and selection of children using EAX
- With reference to [1], the method of **GA-EAX for ATSP** was applied.
- For the selection of AB cycles, "EAX-1AB" from reference [1] was applied.
- For the reconstruction of permutations, **the logits of the parent individuals output from Gemma model were used to select edges** that are estimated to reduce perplexity.
 - Since the perplexity of a sentence is calculated from the logits, by retaining the logits of the words in the sentence, it is possible to roughly estimate the perplexity when a different word is placed next to a certain word.
- For the selection, with reference to "SEL1" from reference [1], the parent individuals were compared with children close to the parent individuals, and if the child individuals were improved over the parent individuals, they were replaced.
- I added other small improvements.
 - When evaluating the scores in the selection process, we imposed a penalty on individuals that were similar to each other to maintain diversity.
 - Individuals that were once eliminated were made to be used again as parents depending on their scores.
 - I tried other crossovers, but almost only EAX worked effectively.

# Discussion
- GA-EAX, which uses the logits that can be output from the language model, was effective in searching for a solution.
- The method presented in [1] appears to be effective in maintaining and optimizing population diversity.

A code sample is shared [here](https://gist.github.com/FujiwaraRyo/9a30b1cc7b615789de0aa264d2e63faa).

# References
[1] Yuichi Nagata; Soler Fernández, D. (2012). A new genetic algorithm for the asymmetric traveling salesman problem. Expert Systems with Applications. 39(10):8947-8953. doi:10.1016/j.eswa.2012.02.029. (https://riunet.upv.es/handle/10251/36442 )
