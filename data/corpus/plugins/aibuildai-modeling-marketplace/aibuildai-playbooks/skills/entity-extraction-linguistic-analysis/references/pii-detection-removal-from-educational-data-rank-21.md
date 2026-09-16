# 23th Simple solution

Competition: pii-detection-removal-from-educational-data
Rank: #21
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497258

Thank you for competition! My solution is simple and no new method, but I will share it.

### Baseline

As a baseline, I mostly used the following Notebook. @emiz6413 Thanks!
https://www.kaggle.com/code/emiz6413/train-deberta-v3-single-model-lb-0-966

All models used DeBERTa-v3-large. I used train+mpware data (use only 30% of negative data).
For the validation, I splitted training data: document % 4 ! = 2 and validation data: document % 4 == 2.

### Ensemble

I made some changes to the training data for ensemble. Ensemble1: added hard samples (samples that DeBERTa predict incorrectly) and used 25% negative data as training data. Ensemble2: Exclude short essay (len_tokens < 100) or negative data (all labels "O") from MPWare data.

### Post-processing

I changed some words ("Mr.", "Dr."...) or not initials not capitalized to "O". 

### not worked for me
   - Data augmentation
   - Longformer-base and Longformer-large
   - Funnel-transformer (Good CV but bad public score...)
   - MLM (Good CV but bad public score)
   - PEFT
   - SiFT
   - RandomMask
