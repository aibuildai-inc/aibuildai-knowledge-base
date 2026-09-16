# 44th Place Solution for the Stanford Ribonanza RNA Folding Competition

Competition: stanford-ribonanza-rna-folding
Rank: #44
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460315

## Context section
- Business context: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/overview
- Data context: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data

## Overview of the Approach

This is my first write-up (also the first time in the medal section 😁). Thanks to the organizers and participants! My approach was based on work we are doing in our lab for RNA secondary structure prediction [1] (this article includes source code). This is a  ResNet-based model that takes the input sequences as one-hot representations and outputs the interaction matrix prediction (similar to the bpp provided). The model has a first 1D feature extraction stage, then it is converted to 2D with a simple matmul, and undergoes a second stage of 2D feature extraction to reach final prediction. For this competition, the output was "flattened" using a sum by columns, thus arriving to a representation of the activation of each nucleotide in the sequence. The inner 2D representation allow us to add bpp information as an additional channel to learn features from.

In terms of data it was rather a simple approach, the best model used the training data with SNR>0.5. Train/test splits were performed using clustered sequences using cdhit-est (80% threshold) [2]. This is important to not overfit patterns on similar sequences. Provided BPP were also used. 




## Details of the submission

I wanted to try different approaches but focused on improving the one detailed above. As training was time consuming, using only the medioids of each sequence cluster proved to reach competitive results (compared to my best submission) thus it was the approach during development. It would be nice to filter leaked test sequences but it seems that my best solution in public LB is the same in private, so good news for the model. Final results were averaging 5 models, 3 using only medioids because I ran out of time. Also didn't use test data for anything except looking at the public LB 

An interesting take is the use of BPP information. As it is based on classical RNA structure prediction methods and do not model pseudoknots, it could affect model prediction in those cases. I trained models without BPP reaching to far worse avg results (public LB ~.16). I did not tried in private yet, but analyzing a sample case as described in [3] it can be seen that the final ensemble model (middle image) miss the predictions pointed by the arrow in the reference prediction (upper image), while the model without BPP (bottom image) have some resemblance. 



Things that didn't work: 
- Tried white noise and flip augmentation
- Using sequences with less than .5 snr
- Tried to use errors during training but didn't reach to a converging method. It could work though 

Hope we see more works on bio sequences!

## Sources
[1] https://www.biorxiv.org/content/10.1101/2023.10.10.561771v1
[2] https://sites.google.com/view/cd-hit
[3] https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/444653
