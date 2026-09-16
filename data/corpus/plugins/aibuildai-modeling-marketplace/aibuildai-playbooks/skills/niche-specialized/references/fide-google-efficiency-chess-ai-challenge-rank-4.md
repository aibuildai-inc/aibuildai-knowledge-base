# 4th place solution

Competition: fide-google-efficiency-chess-ai-challenge
Rank: #4
Source: https://www.kaggle.com/c/fide-google-efficiency-chess-ai-challenge/discussion/563173

Following Kaggle's conventions, I'd like to share my approach.  

It would be great if other teams could also share their approaches in the Discussion!  

My repo is [here](https://github.com/Lgeu/kaggle-stockfish). This write-up was machine-translated, and you can find [the original Japanese version](https://github.com/Lgeu/kaggle-stockfish/blob/main/writeup.md) inside.

Edit (3/30): Added the Kaggle Notebooks section and changed the title from "My approach" to "4th place solution".

# Overview  

I based my approach on Stockfish 16. By adding a small neural network (not NNUE) to the HCE, I was able to improve the Elo rating by around 30 points.  

# Choosing the Base Engine  

After reviewing discussions and messages on Discord, I decided that using HCE could achieve a competitive ranking. For this reason, I selected Stockfish 16, the last version that still supports HCE.  

I didn’t consider engines other than Stockfish. This is because, during my participation in [Hungry Geese](https://www.kaggle.com/competitions/hungry-geese), I researched Shogi AI and learned that it was heavily influenced by Stockfish. This sparked my interest in Stockfish.  

# RAM Usage  

The memory usage breakdown is as follows:  

## Pawn Hash Table: 640KiB  

I reduced the size of some member variables and decreased the number of elements in the hash table from the original 131,072 to 8,192.  

## Continuation History: 512KiB  

I replaced it with a hash table containing 262,144 elements. I didn’t implement collision checks for the hash.  

## Transposition Table: 1MiB  

I kept the smallest possible size allowed in the original Stockfish. This applies to the Pawn Table and Continuation History as well, but I determined the sizes based on intuition and didn’t fine-tune them.  

I also removed the extra memory allocation that the original Stockfish used for Large Pages. (Kaggle’s environment likely doesn’t support Large Pages anyway.)  

## Other: Around 1MiB?  

I removed all unnecessary modules.  

The Magic Bitboard for rooks consumed a lot of memory. However, I found that the Classical Approach from CFish, as described in the [Chess Programming Wiki](https://www.chessprogramming.org/Sliding_Piece_Attacks), seemed like a good alternative, so I replaced it.  

## glibc: Around 3MiB?  

Since the program links to glibc, its Resident Set Size (RSS) should require around 3MB of memory. However, glibc was already loaded in Kaggle’s environment, so it didn’t seem to count toward the 5MB limit.  

On the other hand, libstdc++ was not preloaded. Linking to it increased memory usage, so I rewrote almost all parts of Stockfish that relied on the C++ standard library to avoid linking to libstdc++.  

While it was still possible to submit an agent with libstdc++ linked, removing the dependency reduced the frequency of timeout losses. (The agent I deployed for the last 17 days of the competition still had the libstdc++ link.)  

# Improving the Evaluation Function  

With the optimizations described so far, along with efficient engine usage (such as time management, enabling pondering, and avoiding unnecessary table initialization at the start of the search), I was able to develop an agent that, on average, ranked within the gold medal range.  

(As many participants probably agree,) Stockfish’s code has been refined over many years, so there aren’t many areas left to improve. The only part that seemed viable for modification was the evaluation function, as NNUE was disabled.  

From discussions on Discord, it appeared that the top teams were using NNUE. Building a small NNUE would have been the most straightforward choice, but I thought extending HCE with a neural network would be more interesting, so I went with that approach.  

## Neural Network  

I used a three-layer MLP.  

The original HCE packs two evaluation values—one for the middle game and one for the end game—into a single 32-bit variable and performs calculations on it. I extended this by adding 14 additional 16-bit values, computing them simultaneously using 256-bit registers, and using the results as the first layer’s output. (In hindsight, I should have separated these calculations from HCE. The quantization process became unnecessarily complex.)  

There were 99 input features, so the first layer had 1,400 trainable parameters, including biases (99 × 14 + 14 = 1,400). These values were computed separately for white and black pieces and then combined into a 32-dimensional vector based on the turn.  

This vector was passed through a Clipped ReLU (as used in NNUE), a fully connected 32×32 layer, another Clipped ReLU, and a final fully connected 32×1 layer to produce the output. The total number of trainable parameters was 2,489.  

For training, I set the target values as the difference between the NNUE evaluation and the HCE evaluation and used MSE as the loss function. I also attempted QAT, though I’m not sure how effective it was.  

I ran approximately 70,000 games using `kaggle-environments`, saving positions encountered during search with a 1/8,192 probability for training. The agents used in these games included ones with HCE-based evaluation, NNUE-based evaluation, and my experimental evaluation function. Both training and self-play were conducted entirely within Kaggle Notebooks.  

When using this NN for evaluation during search, simply adding the output to the HCE value resulted in less than a 10 Elo improvement. However, adding only half of the output led to a 30 Elo gain compared to using HCE alone.  

Unfortunately, I only realized this on the final day of the competition, so I didn’t have time to experiment with a larger NN.  

# Submission Strategy  

Since I wasn’t sure whether the NN-based improvements would hold up outside my local environment, I submitted both a [HCE-only agent](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/leaderboard?dialog=episodes-submission-42815406) and an [NN-enhanced agent](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/leaderboard?dialog=episodes-submission-42836820).  

# Kaggle Notebooks

I've made public the Kaggle notebooks used to generate NN weights.

## Building Chess Bots for Generating Training Data

Bots are built to save positions encountered during their search as features with a certain probability. This includes bots using HCE, NNUE, and an evaluation function under development, as described above.

- https://www.kaggle.com/code/nagiss/chess-042-f040-training-data-generator
- https://www.kaggle.com/code/nagiss/chess-052-f042-data-generator-use-hce
- https://www.kaggle.com/code/nagiss/chess-058b-data-generator
- https://www.kaggle.com/code/nagiss/chess-058c-data-generator-060-params

## Generating Training Data

Bots built for data generation play against each other. Since many CPU cores can be used in Kaggle's TPU environment, some notebooks were executed there.

- https://www.kaggle.com/code/nagiss/chess-043-generate-training-data-seed0
- https://www.kaggle.com/code/nagiss/chess-043b-generate-training-data-seed400
- https://www.kaggle.com/code/nagiss/chess-043c-generate-training-data-seed800
- https://www.kaggle.com/code/nagiss/chess-043d-generate-training-data-seed1200
- https://www.kaggle.com/code/nagiss/chess-043e-generate-training-data-seed1600
- https://www.kaggle.com/code/nagiss/chess-043f-generate-training-data-seed2000
- https://www.kaggle.com/code/nagiss/chess-043g-generate-training-data-seed2400
- https://www.kaggle.com/code/nagiss/chess-043h-generate-training-data-seed2800
- https://www.kaggle.com/code/nagiss/chess-043i-generate-training-data-seed3200
- https://www.kaggle.com/code/nagiss/chess-043j-generate-training-data-seed3600
- https://www.kaggle.com/code/nagiss/chess-043k-generate-training-data-seed4000
- https://www.kaggle.com/code/nagiss/chess-043l-generate-training-data-seed4400
- https://www.kaggle.com/code/nagiss/chess-043m-generate-training-data-seed4800
- https://www.kaggle.com/code/nagiss/chess-043n-generate-training-data-seed5200
- https://www.kaggle.com/code/nagiss/chess-043o-generate-training-data-seed5600
- https://www.kaggle.com/code/nagiss/chess-053-f043l-datagen-hce-seed0
- https://www.kaggle.com/code/nagiss/chess-053b-datagen-hce-seed400
- https://www.kaggle.com/code/nagiss/chess-053c-datagen-hce-seed800
- https://www.kaggle.com/code/nagiss/chess-053d-datagen-hce-seed1200
- https://www.kaggle.com/code/nagiss/chess-053e-datagen-hce-seed1600
- https://www.kaggle.com/code/nagiss/chess-053f-datagen-hce-seed2000-7999
- https://www.kaggle.com/code/nagiss/chess-053g-datagen-hce-seed8000-13999
- https://www.kaggle.com/code/nagiss/chess-059-f053g-datagen-058b-vs-hce-seed0
- https://www.kaggle.com/code/nagiss/chess-059b-datagen-058c-vs-hce-seed0

## Organizing Training Data

Since the amount of data is large relative to the model size, data loading tends to become a bottleneck during training. Therefore, the data is organized for efficient loading. To bypass Kaggle notebook output data size limitations, the notebooks were split into eight parts.

- https://www.kaggle.com/code/nagiss/chess-064-f060-split-data-0-8
- https://www.kaggle.com/code/nagiss/chess-064b-f060-split-data-1-8
- https://www.kaggle.com/code/nagiss/chess-064c-f060-split-data-2-8
- https://www.kaggle.com/code/nagiss/chess-064d-f060-split-data-3-8
- https://www.kaggle.com/code/nagiss/chess-064e-f060-split-data-4-8
- https://www.kaggle.com/code/nagiss/chess-064f-f060-split-data-5-8
- https://www.kaggle.com/code/nagiss/chess-064g-f060-split-data-6-8
- https://www.kaggle.com/code/nagiss/chess-064h-f060-split-data-7-8

## Training

Training is performed in Kaggle's GPU environment. The output of the 7th cell was used for the final submission as params.h.

- https://www.kaggle.com/code/nagiss/chess-065d-lr-1e-2-epoch-500
