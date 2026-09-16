# [1st place solution] Transformer model with Dynamic positional encoding + CNN for BPPM features

Competition: stanford-ribonanza-rna-folding
Rank: #1
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460121

The Ribonanza RNA Folding competition has been an amazing opportunity and we are really glad to have participated in it. Kudos to the hosts for their substantial effort in keeping the high organizational level and giving the contestants plenty of valuable insight and support. We would also like to thank the community for the fruitful public discussions and coding initiatives. Congratulations to everyone involved! 🎉

# Code
Open source code is available on [github](https://github.com/autosome-ru/vigg_ribonanza/).

# TLDR

Our solution is based on transformer architecture. We predicted base pair probability matrix(BPPM) for each RNA sequence using EternaFold and added convolutional blocks to our architecture to process BPPM features. For some models we added Squeeze-and-Excitation layer in convolutional blocks Also we add these features with attention values before softmax operation in Self-Attention block. To allow better generalization for longer input we implemented Dynamic Positional Bias. Then we ensembled models with slight differences in architecture and training process.

# Data Preprocessing

At first, for each sequence in train and test datasets we calculated a base pair probability matrix (BPPM) using EternaFold. During training and inference phases we passed to our model the RNA sequences encoded with a learnable embedding layer. Each nucleotide was considered a token and special **< start >** and **< end >** tokens were placed at both ends as well. To provide the model with the information about whether the sequence comes from a “clean” subset of training dataset (which SN_filter values show - 1 corresponds to “clean” sequences with high signal-to-noise ratio and 0 is otherwise) we encoded SN_filter values with a learnable embedding layer and added corresponding  embeddings to sequence embeddings. The embedding dimension was chosen to be of size **192**. BPPMs were padded with zeros at their margins to account for adding **< start >** and **< end >** tokens. The figure below summarizes the data preprocessing part.



# Model

The model idea is partially inspired by @shujun717 solution to the Open Vaccine challenge, [link](https://www.kaggle.com/competitions/stanford-covid-vaccine/discussion/189564)

The model takes a sequence of tokens and BPPM as input and outputs DMS_MaP and 2A3_MaP reactivities for each input nucleotide. Its architecture comprises 12 consecutive Transformer Encoder Layers and an output projection linear layer. Each Transformer block takes a sequence of tokens and BPPM features from the previous layer and outputs updated feature maps as shown below. The Transformer Encoder block adopts common transformer encoder architecture, except we modified the Self-Attention block to ensure interaction between BPPM features and sequence features.



## Self-Attention Block

In the Self-Attention (SA) block we implemented an attention mechanism with the following modifications: after attention values for each head are calculated, we add BPPM features updated by the ‘Convolutional block’, which outputs BPPM features with the number of channels corresponding to the number of heads in the SA block. We set the number of heads in SA and corresponding channels of BPPM features to be 6. Thus, the hidden dimension size of Q, K, V matrices is 32. The overall structure of the SA block is shown below.



## Dynamic Positional Bias

The sequence length is distributed differently in train and test datasets (test sequences are generally longer). To allow for the better generalization of longer inputs we implemented a positional encoding to be added to attention values. We found Dynamic Positional Bias to be of more use compared to other relative positional encoding methods we tried, such as xPos (rotary positional encoding) and ALiBi. Dynamic Positional Bias calculates for each head a relative positional bias map, which is learnable and depends on sequence length. Relative positional bias doesn’t allow to leverage distance from start and end of sequence, so tokens **< start >** and **< end >** were added to fix that.

.png?generation=1701994757603986&alt=media)

## Convolutional Block and SE block

The models in the final ensemble come in two versions that are slightly different in the structure of the Convolutional block. The basic Convolutional block consists of 2D convolutional layer, batchnorm layer, activation and learnable gamma parameters that scale the output feature channels, whereas the modified version of this block (SE-Convolutional block) also contains Squeeze-and-Excitation layer, hence the name. The SE layer applies input-dependent rescaling of values along the channels as shown below. Thus, the only difference between models in the ensemble is the presence of the SE layer.



# Training Process

The training process has been adapted from the [notebook](https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb/notebook) by @IAFOSS
We used a one-cycle learning rate schedule (pct_start=0.05, lr_max=2.5e-3,) coupled with AdamW optimizer (wd=0.05), and batch size set to 128.
The number of epochs for model training was determined from the dataset size.
For final models (trained almost on the whole dataset) we used 270 epochs. 
In each epoch, 1791 batches were processed (we kept this number due to historical reasons), elements of each batch were sampled from dataset with the weight = 0.5 * torch.clamp_min(torch.log(sn + 1.01),0.01)

We have also found that training model with a simple SGD optimizer for ~15 epochs of 500 batches each additionally improves model performance (the number of epochs varies so we used a small validation set to determine the exact number of epochs)

Additionally, one model was trained to predict 2A3 given DMS, RNA sequence and BPPM (dms-to-2a3 model).

# Inference
At the inference phase for all inputs we set the SN filter value to be 1 as if they come from a “clean” dataset.

# Ensembling

We ensembled 15 models with SE-Convolutional block, 10 models with plain Convolutional block, 2 with plain Convolutional block trained on split by sequence lengths (one of these two models accepts bracket features).  We took the average of their predictions. Then we predicted 2a3 reactivities based on averaged dms reactivities using dms-to-2a3 model and added these predictions to averaged 2a3 reactivities in the following way: (27/28)*averaged_2a3 + (1/28)*predicted_2a3.

# Other splits 

The clear problem with a simple KFold split is the high sequence similarity across the training dataset and the fact that test sequences are very distinct from the training data. This might hamper the model development, because the increase in quality on the validation set could be due to overfitting rather than actual improvement.

We have calculated a hamming distance matrix for all the sequences present in the training set and performed a modified DBSCAN clustering procedure with distance threshold set at 0.2. In the following picture we show the clusters mapped to their respective cluster identifier (a cluster ID was assigned as a number of the smallest sequence within that cluster in the train dataset).



We tested several splits based on sequence identity (the easiest one is just to split data into folds without shuffling) and found out that while the model final validation performance degrades as we choose more and more stricter distance threshold, its relative value behave the same way as for a simple KFold split. So, we decided to use simple KFold split than training final models 

Also, we have conducted a test of the model performance on length-based split. For that, we trained a model on the short sequences (length < 206) and validated it on sequences of size = 206. The behavior of the validation metric was slightly noisier but still highly correlated with a metric for a simple KFold split.

# Public data leakage 

Approximately 13% of public test sequences are identical to the ones present in the train dataset (by sequence). To avoid selecting a model that is memorizing more of these sequences rather than learning RNA-related stuff, we zeroed out the predictions for these sequences in most of our submissions, while sometimes sending non-zeroed out submissions in order to compare our performance to other participants.

# Features we also tried

## capR

We don’t have conclusive results for that feature. It seems like it is not beneficial for the model on average but sometimes it resulted in a better model and sometimes – in worse. We decided to not use this feature.

## Brackets

We have tried to use brackets generated by EternaFold, ContraFold, ViennaRNA, etc., as well as programs for pseudoknots prediction (like IPknot). However, after adding EternaFold BPPMs to the model, adding other features yields no significant increase in model performance. For some models we used brackets just to augment model

## Different BPPMs

All other BPPMs (ContraFold, ViennaRNA, RNAsoft, RNAstructure) result in a suboptimal model.
Averaging BPPMs doesn’t result in better performance.
BPPMs from RFold yield the same quality as ViennaRNA BPPMs.
The RNA-FM model produces both per-nucleotide embeddings and BPPM-like matrix. Still, those don’t help the model at all, and using them results only in a slight increase in model performance then compared to sequence only model
 
## SQUARNA matrix

SQUARNA ([github](https://github.com/febos/SQUARNA)) outputs a matrix different from BPPM, but can be used in the same manner. Unfortunately, this feature also didn’t yield any additional performance increase.

# What we also tried

## Fully-Convolutional architecture

The initial reason we decided to take part in the competition was to test our model LegNet [link](https://academic.oup.com/bioinformatics/article/39/8/btad457/7230784), which shows SOTA results on DNA sequences and worked well on some RNA-related tasks (not yet published).
Unfortunately, any modifications of this architecture resulted in subpar performance when compared with properly tuned transformer models. This can be explained by the fact that predicting RNA secondary structure requires attending to long-range contacts. The transformer architecture suits better for such cases.

## Subsetting data 

Subsetting data (filtering by different thresholds on SN ratio) resulted in a performance boost for all models. However, this technique was superseded by weight sampling, which proved itself to be more effective.

## Fine-tuning on public datasets 

We tried to fine-tune our model on a public dataset, gathered by the organizers ([link](https://www.kaggle.com/datasets/rhijudas/rmdb-rna-mapping-database-2023-data/data)) by training model to predict the results of the public experiments (excluding ones with a small number of samples) and for Ribonanza data simultaneously. Unfortunately, this also gave no boost to the model performance.

## Using 3D data 

We tried to use data about predicted 3D structure of 100k sequences from the train dataset but gave up on that once we had visually analyzed them:

 

## Absolute positional embedding

Using absolute positional embedding leads to unsolvable issues when generalizing upon longer sequences.

## Relative positional embedding 

The simplest approach is to augment absolute positional encoding during the training phase to shift randomly from 0 to (Lmax - seqlen) position. This indeed solves the issue with extrapolation, but works worse than other methods.
Rotational positional embedding, unfortunately, doesn’t help the model to generalize on larger lengths.
ALiBi positional embedding solves the issue with extrapolation but even after keeping it only for a part of heads (as suggested in https://github.com/lucidrains/x-transformers) still behaves worse than dynamic positional bias.

## Augmentation 

First, we tried to use reverse augmentation. This can be done in three ways:
reversing sequence before any modifications,
reversing sequence before padding, but after adding <start> and <end> tokens,
reversing sequence after padding.

The first two ways yield no gain for all variants of models we tested. Yet, the third one (upon coupling with additional finetuning) gave us a good result for a model with xpos positional encoding. The resulting single-model performance was 0.13937 on the public leaderboard. Unfortunately, xpos shows rather poor performance on long sequences so we abstained from using this model in the final submission.

We have also tried shift augmentation and different sequence padding approaches. This didn’t improve our model performance as well.

## Sliding window

One of the possible ways to generalize for larger sequences is to predict reactivities using the sliding window. However, the idea is somewhat wrong in a biological sense, and it results in performance degradation when testing on train dataset sequences.

## Pseudolabelling

Once we obtained ensembles of best-performing models we tried to use them to pseudolabel test dataset and use predictions with the highest confidence to train new models. While this indeed results in a better single-performing model, adding such a model to ensemble doesn’t improve ensemble performance.

## Changing loss

Instead of filtering sequences with low SN we tried to mask positions with high reactivity error as it was done by @nullrecurrent in Open Vaccine challenge ([link](https://www.kaggle.com/competitions/stanford-covid-vaccine/discussion/189620))
This resulted in poor performance.
We tried to weight loss for each sequence by its SN – it didn’t result in any improvement

# Tools

[arnie](https://github.com/DasLab/arnie/tree/master)
[EternaFold](https://github.com/eternagame/eternafold)

# Links

## Squeeze-and-excitation block
Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).

## Dynamic positional bias
Wang, W., Chen, W., Qiu, Q., Chen, L., Wu, B., Lin, B., ... & Liu, W. (2023). Crossformer++: A versatile vision transformer hinging on cross-scale attention. arXiv preprint arXiv:2303.06908.
