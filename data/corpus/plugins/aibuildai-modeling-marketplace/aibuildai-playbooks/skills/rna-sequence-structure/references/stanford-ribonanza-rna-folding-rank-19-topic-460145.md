# 19th Solution

Competition: stanford-ribonanza-rna-folding
Rank: #19
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460145

Thank you for host! it's very interesting competition.
I share about my solution.
 
# TL;DR
1. Conv + Transformer + LSTM Architecture
2. Pseudo Labeling
3. Weighted Average

important point is that I use test data while training process for robustness of Private data.

# Solution
## Architecture
basic architecture

1. Conv + Transformer + LSTM
2. Conv + LSTM + Graph Attetion

## Training Strategy
1. MLM
2. Traning
3. Pseudo Labeling

### MLM
I use first pretreind Masked Language Model(Masked RNA Model?)
I delete some RNA sequence sometimes(0.3), and predict deleted sequence.

### Training
#### Feature
1. sequence
2. predicted_loop(eternafold/contrafold_2/vienna_2)
3. structure(eternafold/contrafold_2/vienna_2)

I use one hot encoding with 2/3 feature, also I express probability(eternafold: (, contrafold_2: (, vienna_2:) ->(:0.66, ):0.33)

#### Training Strategy
Training is very simple
1. use Filtered Dataset
2. weighted loss(sequence)
3. remove some sequence(structure/predicted_loop)

### Pseudo Labeling
Private dataset is longer than Public dataset. I decide to get robustness from test data.
but this competiton is regression, I cannot get confidence.
I use pretrain, then train only training dataset.

I apply iterative that pseudo labeling.
also few step later, I relabel noisy training data(=filter0), and train this process with pseudo labeling data.

Public: 0.141->0.139->0.1384->0.1383

## Ensemble
use weighted average
I check only public because I didn't believe cv score when I used pseudo labeling.

## Note
I forgot exhanging old and new dataset....
it's very important...(Single model achieve Private 0.143..)
