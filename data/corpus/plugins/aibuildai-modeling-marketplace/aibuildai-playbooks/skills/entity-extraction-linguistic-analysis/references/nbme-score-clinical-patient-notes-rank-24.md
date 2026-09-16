# [24th place solution]: Pretrain tasks is the key!

Competition: nbme-score-clinical-patient-notes
Rank: #24
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323168

First, thanks Kaggle and the competition hosts for a challenging competition in NLP after Feedback Prize - Evaluating Student Writing ended. This summarizes the effort of our team and it was a great experience working with my teammates ! This is a writeup of our collective team effort. We don't have much time for NBME challenge (about 3 weeks) after focusing on the Happywhale competition.

After reading solutions by top teams we realized what was missing from the team was the pseudo label (actually noticed before, we just didn't make it work like other teams did) 

# Modeling
## Pretrain task
1. Task adaptation with MLM (mask language  pre-training (used standard masked language modelling for pre-training with token masking probability of 0.15): Deberta-V2-xlarge, Deberta-V2-xxlarge and Deberta-V3-large
2. Task adaptation with WWM (whole word masking) pre-training: Deberta-V2-xlarge and Deberta-V3-large
## Model Experience
We try a lot of models and custom heads:
- Deberta-V3-large
- Deberta-V2-xlarge
- Deberta-V2-xxlarge
- Roberta-large
- Deberta-V3-large with LSTM/GRU head
- Deberta-V3-large with custom head (CNN-1D)
- Deberta-V3-large concat last 4/12 layers

Some of the results are detailed in the photos below
[[nbmemodel.jpg]](https://postimg.cc/XBMMv8z4)
#Ensembling
Simple average is used by us for final ensemble solution, maybe weighted average would be better but we didn't try. I also want to try Stacking but don't have time.
#Post Processing
Our post process method consists of 2 parts:
- One that is almost the same as the Misaling annotations like 11th place solution
- One is the same as the threshold of the 4th place solution, our team uses the threshold for each case num and feature num, CV increased from 0.895 to 0.896 but LB decreased a bit, so we stopped to range 0.45-0.55 (this range threshold). And we tune based on CV to reduce overfit on public LB, we use score increment on both CV and LB. The results of post processing on LB (both public and private) increased by about 0.001. Our rankings haven't changed much on public and private LB :D
#Thing that work but not have time to try
1. Meta Pseudo Labels (MPL)
2. Knowledge Distillation (KD)
3. Mixup, Random mask
# What didn't work for our team
1. Multisample Dropout
2. Pseudo labeling
3. LSTM/GRU head
