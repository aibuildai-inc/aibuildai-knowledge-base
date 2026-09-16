# Niboshi's solution

Competition: fide-google-efficiency-chess-ai-challenge
Rank: #10
Source: https://www.kaggle.com/c/fide-google-efficiency-chess-ai-challenge/discussion/563866

We first went with Stockfish 4, then Stockfish 16, then finally Cfish due to RAM constraint.

Our code is [here](https://github.com/chettub/Niboshi).

# NNUE

We disabled HCE and relied our evaluation totally on NNUE.

We assumed that the accuracy of the evaluation function was essential, and therefore searched for a binary-size-efficient network.

Our final network was inspired by the latest Stockfish's network. To save the size, we replaced the feature transformer by concatting the output of the following 2 networks. The first is a CNN with input channel 12 (=number of pieces), output channel 13, kernel size 15x15, and padding 7, so that the CNN could produce features for each square. The second is a dense network with 64 outputs, but it shares the same weight for neighbor positions. A normal dense network would have 768 inputs, but ours has only 96 inputs.

However, one thing that we missed was that the computation cost was very heavy; the equivalent L1 size was 896, too large to compensate the accuracy loss caused by the shared weights in the CNN.

As for the training dataset, we used a mix of leela-generated data and stockfish-generated data, though we didn't see much difference when changing the dataset.

# Other changes

- Transposition table : 512 KiB
- Use hash table for continuation History : 128 KiB
- Compile nnue.c with -O3, and others with -Os
- Delete features that were unused, or did not hurt the performance too much

# Final remarks

We were happy to be able to participate in this competition.
I personally knew little about chess programming before entering this competition, but I learned a lot from the competition that I could not have learned from other places.

Congratulations for the top 3 teams. You chess engine developers totally deserve it.

Many thanks to the organizers and the participants. Thank you @bovard, [for taking action when it was necessary and only when absolutely necessary](https://www.kaggle.com/competitions/llm-20-questions/discussion/531387). And special thanks for those who worked hard debugging the environment.
