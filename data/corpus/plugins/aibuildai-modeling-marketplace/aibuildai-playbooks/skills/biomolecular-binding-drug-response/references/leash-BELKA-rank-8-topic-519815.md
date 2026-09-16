# 8th place private solution - a single SMILES-based transformer model is all we need. And a proper benchmarking

Competition: leash-BELKA
Rank: #8
Source: https://www.kaggle.com/c/leash-BELKA/discussion/519815

# Context
Business context: https://www.kaggle.com/competitions/leash-BELKA/overview  
Data context: https://www.kaggle.com/competitions/leash-BELKA/data

# Summary


A summary of the solution: ligand-based, SMILES-only, multi-target transformer model trained independently on shared and non-shared building blocks splits. A special hit identification data splitting technique, and k-fold training and validation are used for non-shared blocks. The models are independently averaged for the shared and non-shared parts of the test set.

# Data splits
This competition hinges on the fact that similar molecules tend to have similar properties - this is the main assumption in lead optimization of initially identified hits. However, the correct evaluation strategy is rarely applied, which is why [Polaris](https://polarishub.io/), a collective effort to establish unbiased benchmarks for drug discovery, was launched recently. Moreover, it's hard to say what is correct for a specific task. Yet, it was shown recently by my colleague Simon Steshin that a splitting strategy by 0.4 Tanimoto similarity should be considered for novel hits identification ([Lo-Hi: Practical ML Drug Discovery Benchmark](https://arxiv.org/abs/2310.06399)). Perhaps we can apply a similar approach to build a more generalizable model in other tasks, particularly in this competition where 66% of the private score is dedicated to the out-of-distribution molecules.

## Non-shared BBs
The molecules provided by the organizers consist of three building blocks, so we may consider splitting by the similarity of building blocks. Let’s examine the distribution of test building blocks to training building blocks:

[Distribution of Maximum Tanimoto Similarities Between Test and Training Building Blocks]

There are many building blocks shared between train and test sets (Max Tanimoto is 1.0), and we obviously need to avoid the same occurrence in our non-shared train/valid split. We see that BB1 doesn't intersect with BB2, and BB2 significantly intersects with BB3. Besides the clear duplicates, there are many similar building blocks, which, as per the above assumption, will introduce a strong bias into our evaluation.

### Hit identification split by building blocks (HIBB)
So, let's eliminate this bias. We will use the Hi splitting algorithm from the Lo-Hi benchmark which solves a Balanced Vertex Minimum k-Cut problem to construct such training and validation sets that the closest molecules between the two sets will be at least the cutoff Tanimoto distance apart ([code](https://github.com/SteshinSS/lohi_splitter)). To decide what threshold to use, let's look at the training set's Tanimoto similarity distributions per block position, considering only the same positions (we will deal with BB2-BB3 cross-positions later):

[Distribution of Maximum Tanimoto Similarities Between Training Building Blocks To Each Other (excl. the same)]

It's an open question of what threshold to set, and most likely, using multiple ones is the best approach. I chose the thresholds close to the mean of the per-position similarities (don't ask me why): 0.7 for BB1<->BB2 and 0.4 for (BB2+BB3<->BB2+BB3). Yes, I merged BB2 with BB3, as they are known to intersect each other. Ok, this split is the hardest one. Training models on it will quickly show that nothing is working (in other words, structure similarity drives a lot of correlation with activity).

[HIBB split: 0.4 Tanimoto dissimilarity split by building blocks]

Pay attention to the exclusivity of building blocks - if a block appears in one set, it cannot appear in another set in any possible combination.

### Weighted building blocks splits (BB)
Although the Hi split is useful, it might be over-pessimistic for the competition's problem, and it removes a lot of data due to the exclusive assignment of building blocks to the train/valid sets. So, I decided to create a simpler split close to what other participants did when replicating the host's split but different in a few aspects:
- The blocks are again exclusive to the train/valid sets
- There's an increased amount of blocks at each position by 3 times compared to the BB split from the great [notebook](https://www.kaggle.com/code/thedrcat/belka-split-cv-like-the-host/) of @thedrcat and work of both @roberthatch and @hengck23 [here](https://www.kaggle.com/competitions/leash-BELKA/discussion/496576)
- There are rolling 5 folds where, for each fold, different blocks are selected at each position
- Training data consists of a mix of `any positive` molecules (the ones that bind to at least one target) and the same amount of `all negative` molecules (no binding to all targets) randomly sampled from the corresponding fold's building blocks
- Validation data consists of a mix of what's left from the `any positive` set and an amount of `all negative` compounds to approximately match the imbalance of the original training dataset, i.e., `any positive` rate is ~1.5%

Overall, the folds look like this (randomization not taken into account for simplicity):
|  Fold | BB1 | BB2&BB3 | BB3\BB2 | train size | train pos rate | valid size | valid pos rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0-50 | 0-101 | 0-5 | 1 875 494 | 50% | 356 620 | 1.67%
| 2 | 51-101 | 102-203 | 6-11 | 1 777 354 | 50% | 406 629 | 1.62%
| 3 | 102-152 | 204-305 | 12-17 | 1 430 522 | 50% | 363 503 | 2.96%
| 4 | 153-203 | 306-407 | 18-23 | 1 924 372 | 50% | 311 943 | 1.47%
| 5 | 204-254 | 408-509 | 24-29 | 1 903 992 | 50% | 313 838 | 1.47%

As with the HIBB split, the validation set is of similar size per fold, but we now have much more training data:

[BB split: weighted building blocks splits]

## Shared BBs
The strategy for the building blocks shared between train and test sets is clear—we just need to overfit to the known building blocks while covering as much data as possible. So, here, I used all 98M molecules, of which I randomly selected 1% for validation and 99% for training.

# Models and training
To put it simply, the training strategies I used are the following:
- **shared BBs**: overfitting while tracking performance on validation, taking the last checkpoint
- **non-shared BBs**: taking the best checkpoint on the validation set (the same strategy for both HIBB and BB splits)

I tried A LOT of models and performed extensive hyperparameters tuning (especially on the non-shared BBs) for the following models:
- [MolFormer](https://huggingface.co/ibm/MoLFormer-XL-both-10pct), [RoBERTa ZINC](https://huggingface.co/entropy/roberta_zinc_480m), [ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR): all showed similar quality with MolFormer and a custom RoBERTa performing better in an ensemble
- GPS++: SOTA full-attention GNN with atoms and bonds features (impl. adopted from [Graphium](https://github.com/datamol-io/graphium)). For atoms, the following features are used: atomic-number, group, period, total-valence, degree, formal-charge, radical-electron, hybridization, chirality, implicit-valence, num_h_atoms, aromatic, in-ring, electronegativity. For bonds: bond-type-onehot, stereo, in-ring
- MPNN++: GNN with the same atoms and bonds features (impl. adopted from [Graphium](https://github.com/datamol-io/graphium)), see performance in the [scaling GNNs paper](https://arxiv.org/abs/2404.11568). In my case, MPNN++ also performed substantially better than GPS++, potentially due to not attending everything to everything, thus having a stronger inductive bias
- Tanimoto similarity scores of building blocks to the top-50 closest molecules put together in a multi-head attention model where queries/keys are the similarities and values are the ground truth values. It was supposed that this kind of a model (I call it the `SimAttn` model) could learn from the closest ranked list of molecules, and it did, but I wasn't able to reach high enough AP with it
- XGBoost on RDKit 210 descriptors normalized with standard scaler and ECFP4
- MLP on RDKit + ECFP4 + similarity features
- Meta MLP model on RDKit + ECFP4 + similarity feats + best performing MolFormer: slightly improved performance on BB split, but not enough to convince me to submit

From the above, the best performing models are the fine-tuned MolFormer (45M parameters) and a custom RoBERTa (~8M parameters) as described in [my company's recent paper](https://arxiv.org/abs/2406.14572). The customization is mainly related to the 500-sized BPE SMILES tokenizer, 15% of masked tokens in 30% of cases, and SMILES re-enumeration in 50% of cases.

# Conclusion
My take on generalizability to unseen chemical subspaces:
1. This isn't a solved problem, not only for this competition but everywhere in the public domain
2. We can address the issue with extensive benchmarking
3. SMILES-based transformer models perform slightly better than the SOTA GNN models
4. Scaling number of parameters **improves** the quality on out-of-distribution molecules
5. Scaling SMILES-based transformers is way easier than GNNs due to the absence of pre-processing, so scaling to billion-size models and datasets to improve generalizability should be considered in the follow-up research

# Acknowledgements
I thank Copilot and Continue.dev for being my coding teammates all the time. I also acknowledge my colleague Simon for his public work on molecule benchmarking. I thank other participants for their solutions and discussions; I didn't find time to contribute to the discussions, but I was aligned with many of them. Huge thanks to Leash Bio and Kaggle for organizing the competition. And I'm sorry for the teams shuffled on the private leaderboard. I had a solution for 0.299 ten days before the end of the competition, so I expect the same to happen to many other teams. I believe they all tried hard to crack this unsolved but intriguing small molecules generalizability problem.
