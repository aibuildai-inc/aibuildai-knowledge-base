# 22nd Place Solution - Z3 Constraint Satisfaction Solver

Competition: conways-reverse-game-of-life-2020
Rank: #22
Source: https://www.kaggle.com/c/conways-reverse-game-of-life-2020/discussion/200608

My approach to this competition was to try a variety of different techniques and publically publish all my ongoing research public throughout the competition, which resulted in a small tail of "ensemble of public notebook" behind me on the leaderboard.

# Results

| score | duration  | cores | name |
| --- | --- | --- | --- |
| 0.08410 | 5000h | 4 | [Z3 Constraint Satisfaction](https://www.kaggle.com/jamesmcguigan/game-of-life-z3-constraint-satisfaction) |
| 0.08631 | 8.25h | 4 | [Image Segmentation Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver)|
| 0.14502 | 100h | 4 | [Hashmap Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver) | 
| 0.14689 | 0.5h | 4 | [Random Forest](https://www.kaggle.com/jamesmcguigan/reverse-game-of-life-decision-tree?scriptVersionId=46840547) |
| 0.14689 | 0.5h | GPU | [Recursive CNN](https://www.kaggle.com/jamesmcguigan/reverse-game-of-life-rnn) |
| 0.31613 | 1.5h | GPU | [OuroborosLife Function Reversal GAN](https://www.kaggle.com/jamesmcguigan/ouroboroslife-function-reversal-gan) | 

# Interactive Playable Game

I had previously written an interactive playable demo of the forward version of this game in React Javascript:
- https://life.jamesmcguigan.com/


# Z3 Constraint Satisfaction
- https://www.kaggle.com/jamesmcguigan/game-of-life-z3-constraint-satisfaction

My first attempt was to use the Z3 SAT solver. 

Design-wise this was implemented as a 3d grid of cells going back delta steps. Game of Life forward rules were added as constraints, along with the stop board position and a no empty board constraint.

As a performance optimization, an additional constraint was added that any dead cell with no neighbors was also dead in the previous time step. This helped prevent an infinite number of zero-point energy solutions of cells from appearing and disappearing in a sea of whitespace. However, this caused unsat on a few boards which required a distance of 2 whitespace. Z3 push and pop were used to check for the distance=1 constraint, then replace it with distance=2 upon unsat.

I also discovered that using boolean rather than integer logic resulted in a 2-4x performance speedup.

The main limitation of this approach was CPU runtime. Larger deltas and high cell counts (with less whitespace) would take exponentially longer. Even with pathos multiprocessing using all 4 CPUs it still took 20,000h of CPU runtime to reach a score of 0.08410 = 29,121/50,000 = 58%. 

Getting 20,000h of runtime on Kaggle required cluster compute with a dataset reimport loop. The codebase was hosted on github, and 9 forks of the notebook were created. The dataset divided by modulo % 9 on the index id. Each notebook fork imported the others as a datasource and the CSV files could be recombined, sorting to remove zeroed entries and awk to ensure unique indexes
using:

```
!find ./ ../input/ -name 'submission.csv' | xargs cat | sort -nr | uniq | awk -F',' '!a[$1]++' | sort -n > ./submission.csv
``` 

Many boards could not be individually solved within the 9h kaggle notebook time limit, so this also required logging timeout errors and skipping them on future runs.

This process took quite a bit of effort, as I would need to manually rerun 9 notebook forks 2-3 timer per day to reach version 135 of the primary notebook. My estimate is that I used about 5000h x 4 CPUs of Kaggle resources on this competition.

I did briefly experiment with making the stop position an optimization goal rather than a constraint but after the first execution kept running into memory and timeout errors.

# Image Segmentation and Hashmap Solvers
- [Summable Primes](https://www.kaggle.com/jamesmcguigan/summable-primes)
- [Geometric Invariant Hash Functions](https://www.kaggle.com/jamesmcguigan/geometric-invariant-hash-functions)
- [Game of Life - Repeating Patterns](https://www.kaggle.com/jamesmcguigan/game-of-life-repeating-patterns)
- [Game of Life - Hashmap Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-hashmap-solver)
- [Game of Life - Image Segmentation Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver)

My [Image Segmentation Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver/) which was inspired by my previous work on the [ARC Abstraction and Reasoning Corpus Geometry Solvers](https://www.kaggle.com/jamesmcguigan/arc-geometry-solvers)

Scores are:
- 0.08549 = 28,731/50,000 = 57.4% [Z3 Constraint Satisfaction Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-z3-constraint-satisfaction/) (still running)
- 0.08631 = 28,446/50,000 = 56.9% [Image Segmentation Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver/)
- 0.14502 = 6,734/50,000 = 13.5% [Hashmap Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-hashmap-solver/)

The idea is that instead of trying to solve the reverse function, we can simply use forward play over millions of boards to generate a database of known 4d shapes, then create a solver using dictionary lookup based on hash key and delta.

At first, you might consider the state space for a 25x25 game of life board might be too large (2^625 = 1.3*10^188), but it possible to exploit rotational, mirror, and translation symmetry to significantly reduce this. For this I developed [Geometrically Invariant Hash Functions](https://www.kaggle.com/jamesmcguigan/geometric-invariant-hash-functions). They work by taking a pixel-wise view of the board, extending out from each pixel in concentric circles, and using [Summable Primes](https://www.kaggle.com/jamesmcguigan/summable-primes) to multiply the count the number of neighbouring pixels at each distance. The result of this is that the hash function remains unchanged after `np.roll()`, `np.flip()` and `np.rot90()` operations.

It is also possible to create a translation-insensitive variant of this hash function that only remains unchanged by `np.roll()` but is affected by `np.flip()` and `np.rot90()`. 

For the test dataset, we hash the board and look it up in the database. If a match is found, then the problem is reduced to a much simpler geometry problem. `np.roll()` and `np.flip()` can be solved by brute force until the second translation-insensitive hash function aligns. `np.roll()` can be solved by taking `np.bincount()` of both axis, and repeatedly rolling it until they align. The result of which is saved in a function that apply the same operation to both the -delta history from the database.

With enough runtime over millions of boards, this approach was able to achieve 13.5% success rate.

This approach is wasteful as many boards contain small self-contained [Repeating Patterns](https://www.kaggle.com/jamesmcguigan/game-of-life-repeating-patterns) with known timelines. If two or more of these shapes appear on the board, then the hashmap database would need to contain every possible permutation of combinations at every distance. 

Image segmentation simplifies this by being able to extract out such self-contained shapes and then solve them individually. To make `skimage.measure.label()` work with diagonal neighbours, it needs to be combined with both `scipy.ndimage.convolve()` as well as tessellation/de-tessellation to catch wraparound shapes on the image border.

This approach, however, results in a respectable 56.9% accuracy and is mostly stumped by the high cell count boards, which it often reads as a single large cluster of cells that most likely has not seen before.

The [Image Segmentation Solver](https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver/) also executes within a single notebook run, which is much quicker than my previous 10,000+ CPU hour Z3 solution.

# Neural Networks - Forward Play
- [Pytorch Game of Life - First Attempt](https://www.kaggle.com/jamesmcguigan/pytorch-game-of-life-first-attempt)
- [Pytorch Game of Life - Hardcoding Network Weights](https://www.kaggle.com/jamesmcguigan/pytorch-game-of-life-hardcoding-network-weights)
- [Its Easy for Neural Networks To Learn Game of Life](https://www.kaggle.com/jamesmcguigan/its-easy-for-neural-networks-to-learn-game-of-life)

I was able to fully solve the forward game of life problem using a neural network, and managed to train my first network to 100% accuracy. After reading the scientific paper [It's Hard for Neural Networks To Learn the Game of Life](https://arxiv.org/abs/2009.01398) I wrote a tutorial on how to hardcode neural network weights and then wrote a response to their paper.

Small neural networks are often hard or unreliable to train because there is limited dimensionality and it is easy to get stuck in a saddle point between different solutions. However I discovered that this effect is reversed if you create a neural network that has only a single mathematical solution. Such is the case with a 10 neuron network (9 inputs + 1 output) which trains with a 90% success rate compared to the 60% success rate to the 12 neuron network (9 inputs + 2 middle + 1 output) which actually has 3 mathematical solutions.  

# Neural Networks - Reverse Problem
- [OuroborosLife - Function Reversal GAN](https://www.kaggle.com/jamesmcguigan/ouroboroslife-function-reversal-gan)
- [Reverse Game of Life RNN](https://www.kaggle.com/jamesmcguigan/reverse-game-of-life-rnn)
- [Reverse Game of Life Inception CNN](https://www.kaggle.com/jamesmcguigan/reverse-game-of-life-inception-cnn)

I experimented with lots of different architectures and learned a lot about neural network design, however, this was somewhat frustrating as whilst I could see some shape patterns start to converge, I never achieved any high scores solving the reverse problem using this approach.

Common problems included having the network converge on solutions such as all 0s, all 1s or all ~0.5s.

At one point I did see 99% accuracy reported in log files training on a 5x5 board, but I forgot to save the model file and was never able to reproduce the results. Maybe this was a case of lottery ticket initialization.

I had high hopes for my Ouroboros Function Reversal GAN. The network was designed to have multiple heads predicting past, present and future for a given input board state. The loss function then passed the output of the Past back into the network to compare accuracy against it's predicted Future. However, it scored 0.31613 on the leaderboard, which is twice as bad as the sample submission. 

Inception CNN combined convolutions of multiple sizes, along with a "count neighbors" convolutions with hardcoded weights. The RNN CNN had the network output an additional state vector, which could then be fed back into the network along with forward play of its output. 

Still unsure if neural networks are just fundamentally unsuited to this type of problem, or if I was doing something obviously wrong in my neural network design.


# XGboost, Random Forest and Decision Trees
- https://www.kaggle.com/jamesmcguigan/reverse-game-of-life-decision-tree?scriptVersionId=46840547

I was quite impressed with what XGboost was able to do on the Abstraction and Reasoning Challenge.
- https://www.kaggle.com/jamesmcguigan/arc-oo-framework-xgboost-multimodel-solvers

Unfortunately, XGBoost doesn't support multi-class outputs, so I switched it out for a Random Forest.

The first nieve attempt of just giving it an entire board and asking it to predict the output scored 0.14689 which was better than the sample submission. 

I suspect the main problem here is that a Random Forest has no knowledge of rotational and translational symmetry, thus is dealing with very sparse training data for when a specific pattern has appeared in a given location.

I thought about this for a bit, and came up with another idea. Reformat the dataset to give a pixelwise view of the board, N nearest neighbors for each cell, and then try to predict each cell individually. The prediction could then be run through forward play and fed back into the network (much like the RNN CNN design).

I didn't get very far with this approach, mostly because it exceeded the 9 hour time limit for notebook execution, even when I replaced Random Forest with a much quicker Decision Tree and limited the dataset only to delta=1. Never got to submit this approach.

# Summary

In theory, the Z3 solver should have been able to get a perfect score, but the runtime was far too excessive, some boards would have individually taken longer than 9 hours to solve, and I ran into my fair share of notebook timeouts and out of memory errors. 

Image Segmentation Solver scored an almost identical score, which mostly comprised the "easy" boards with small self-contained shapes. Geometrically Invariant Hash functions and Summable Primes were new mathematical inventions.

Neural Networks could easily solve the forward problem, and this was great training in neural network design and implementation but wasn't able to make much progress on the reverse problem using this approach.

Decision Trees didn't get fully explored.

As I understand it, most of the top entries on the leaderboard were written in C, as was also the case in the ConnectX competition, so I probably need to start learning Rust if I want to be competitive on Kaggle.

Overall I learned a huge amount during this competition and had lots of fun.
