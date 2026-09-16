# 3rd Place Solution: Meta Pseudo Labels + Knowledge Distillation

Competition: nbme-score-clinical-patient-notes
Rank: #3
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322832

First of all, thank you NBME and Kaggle for hosting such an amazing competition. It has been an enriching experience with a steep learning curve. My sincere thanks to the community for sharing great ideas and engaging discussions.

## Code
* https://github.com/rbiswasfc/kaggle-nbme-3rd-place-solution

## Summary

- Task adaptation with MLM pre-training
- Multi label token classification (Predicting Inside, Beginning and End of answer spans)
- Training Diversity
  - Meta Pseudo Labels
  - Knowledge Distillation from Model Ensemble
  - Marked tokens: add a prefix token to feature text e.g.: ["QA CASE=0"]
  - Stochastic Weight Averaging (SWA)
- Blending: weighted average of character level predictions from different models
- Post Processing: time duration related features e.g. for `feature 309: duration-2-months`  filter out spans containing `2 weeks`, `years` in it etc

## Models
- DeBERTa Large
- DeBERTa XLarge
- DeBERTa V2 XLarge
- DeBERTa V3 Large

## Workflow

### 1. Task Adaptation

I continued pre-training of all DeBERTa models using the patient notes. Used standard masked language modelling for pre-training with token masking probability of 0.2. In the subsequent training steps, task-adapted models are used as backbone.

- References:
  - Don't Stop Pretraining: Adapt Language Models to Domains and Tasks (https://arxiv.org/abs/2004.10964)

### 2. Meta Pseudo Labels (MPL)

I felt the key to doing well in this competition was to make most out of the unlabelled data. To this end, I experimented with (1) standard Pseudo Labelling, (2) Weak Supervision: Contrastive-Regularized Self-Training Approach (https://arxiv.org/pdf/2010.07835.pdf) and (3) Meta Pseudo Labels (MPL). Among these, MPL worked the best. Thanks to @hengck23 for sharing this excellent notebook on MPL: https://www.kaggle.com/code/hengck23/playground-for-meta-pseudo-label. I adapted this notebook for NBME token classification task. I used both soft & hard pseudo labels for diversity during different training runs.

- References:
  - Meta Pseudo Labels (https://arxiv.org/abs/2003.10580)
  - Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach (https://arxiv.org/pdf/2010.07835.pdf)

### 3. Fine-tuning of Student

During MPL, the student model is trained only on unlabelled data using pseudo labels from teacher. So the student can be further fine-tuned on actual training data for additional performance boost. During fine-tuning, I also used Stochastic Weight Averaging (SWA) for better generalization.

### 4. Knowledge Distillation (KD)

For DeBERTa Large, I trained two models using MPL (each with 180k randomly sampled unlabelled data + all labelled data). Next I used these two models as teacher to distill their combined knowledge (mean token predictions) into another DeBERTa Large model. I used the following loss during distillation: `0.15 * BCE Loss on Labelled Data (actual annotations) + 0.15 * BCE Loss on Labeled Data (pseudo labels from teaches) + 0.7 * BCE Loss on Unlabelled Data with Pseudo Labels`. Rational for using KD: In literature, there are examples where student outperforms teachers specially when additional data is involved. Furthermore, it adds to model diversity.

References:

- Can Students Outperform Teachers in Knowledge Distillation Based Model Comparison? (https://openreview.net/pdf?id=XZDeL25T12l)

## Marked Tokens

This basically involves modifying the feature text with a prefix token. I created 10 new tokens for each case e.g. "[QA CASE=0]". I thought some of the feature texts from different cases are very close to each other e.g. Feature 314: nausea, Feature 508: Associated-nausea. Therefore I felt, giving additional context of case number in feature text would help the model to distinguish between these cases.

## Additional Details

- For MPL training, both Teacher and Student models need to be loaded into memory. To avoid memory overflow issues, I used the usual tricks such as mixed precision training, 8-bit adam optimizer, batch size of 4, freezing of lower layers & gradient checkpointing.
- In ensemble, majority of the models are trained using all data instead of individual folds
- For token classification head, concatenated hidden outputs from last 12 transformer layers.
- Reinitialized last 4-12 transformer layers
- LR Scheduler: cosine schedule with warmup
- Multi-label classification: 3 labels -> [token is inside answer span, token is start of answer span, token is end of answer span].

## Things that didn't work

- Weighted Box Fusion (WBF): With WBF, the LB scores decreased by around 0.002
- Non Maximum Suppression (NMS): simple blending of char probs worked the best
- Tuning of model weights
- Tuning of thresholds (used 0.5 for all features)
- Most of my post-processing attempts
