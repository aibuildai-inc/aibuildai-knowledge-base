# 5th Place Solution

Competition: stanford-rna-3d-folding
Rank: #5
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/5th-place-solution

# Stanford RNA 3D Folding 5th Place Solution
First of all, I never thought we would be able to finish with such a great result, placing 5th.

I'd like to thank hengck23 for his excellent discussions throughout the competition, lihaoweicvch for sharing his fine-tuning code, and everyone else who shared their insights in the discussion thread. And most importantly, I'd like to thank the hosts and Kaggle for providing us with this fantastic competition.

# Context
- Business context: [https://www.kaggle.com/competitions/stanford-rna-3d-folding/overview](url)
- Data context: [https://www.kaggle.com/competitions/stanford-rna-3d-folding/data](url)

# Overview of the approach
All submitted models are ensembles of Protenix fine-tuned models.
Fine-tuning was carried out along two main lines.
1. Curriculum learning based on Protenix's technical report[1].
2. Fine-tuning using clustered RNA.

# Detailed solution
- Data used
Data in train_sequences.csv and train_sequences.v2.csv that contain MSA, and their label data.

- Fine-tuning with RNA-MSA support
We fine-tuned a forked version of the Protenix model[2], modified to accept multiple sequence alignments (MSA) as input.

- Cut-off date filtering
We applied two cut-off dates to curate the training data and reduce potential data leakage:
*2021-09-30*: To filter out any data that might have been included in Protenix’s original pretraining, based on its technical documentation[1].
*2024-05-16*: To ensure only data excluded from the public leaderboard was used, following host guidance[3].

## koooeo Part
- Sequence length-based partitioning
Training data was divided into groups by sequence length to enable more stable and tailored model training per group.

- Curriculum-style fine-tuning
We adopted a step-wise fine-tuning strategy, first training on shorter sequences and progressively moving to longer ones.



## Shun Kuraishi Part
- Clustering and representative fine-tuning
To improve training efficiency and structural diversity, we clustered the training data into 18 groups based on sequence and structural features.From each cluster, we selected 2-4 representative sequences for further fine-tuning.


- Sequence length gap pattern learning
After segmenting the data into sequence-length bands, we trained a dedicated model for each band (e.g., 0–300, 300–500, 500–700, etc) so that each model specialized on its assigned range.

- Specialized model assignment by sequence range
We trained separate models for different sequence length bands (e.g., 0–300, 300–500, 500+),and at inference time, predictions were made using the model specialized for each input's length category.

- Ensemble weighting based on public LB performance
We found that the model trained on the 208–300 range performed particularly well on the public leaderboard.During inference, we performed a 2:3 weighted ensemble between this strong model and the model assigned to each sequence length band.

# What went well
- We believe that simply passing the data with MSA directly into the model did not improve performance, but dividing the data by sequence length and by clustering may have enabled more effective training. By narrowing down the data, we think it may have helped correct biases in the distribution of sequence lengths and clusters.

# What didn't work
- We were unable to find a method to select the optimal structure among the outputs of multiple models.
In the end, we adopted an approach of assigning the outputs of different models to five predictions for each target.
- We could not successfully evaluate which model was optimal.
Although we evaluated model performance using data that was not used for training, we found no correlation between those local scores and the public LB scores. Likewise, there was no correlation between the public LB scores and the local scores. I am curious about how others evaluated their models.

# Sources
[1] [https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf](url)
[2] [https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/573495](url)
[3] [https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/572096#3173463](url)
