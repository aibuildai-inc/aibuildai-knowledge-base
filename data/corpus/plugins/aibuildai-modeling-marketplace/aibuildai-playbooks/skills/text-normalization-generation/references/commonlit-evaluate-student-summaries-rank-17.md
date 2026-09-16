# 17th Place Solution

Competition: commonlit-evaluate-student-summaries
Rank: #17
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446648

Congrats to all the winners, and thanks to organizers and all participants. 
This is my first NLP competition and I have learnt a lot to deal with this area.

**Overview**
Weighted average of 4 models:
1. deberta-v3-large. Using all of title, quesiton, prompt_text and summary. GroupKFold with prompt_id. (CV:0.488, Public:0.430, Private:0.463)
2. Same as model1, but using mask augmentation and trained on all data.
3. Same as model2, but using mask augmentation and trained on all data. (Only seed is different from model2)
4. LGBM trained from the output of debereta-v3-large (prompt_text is not used for this model) and handcraft features.

Ensemble Score: Public:0.422, Private:0.459



**Training Settings (model2,3)**
- Epoch: 3
- Loss: SmoothL1Loss
- lr: 1e-5
- Optimizer: Adam (weight_decay: 5e-4, beta=(0.9, 0.999))
- Scheduler: cosine scheduler
- token length: 2024 (training: 2024, inference: 1664)

**Did work**
- Mask augmentation
  - Mask augmentation doesn't improve cv, but loss curve became more stable. I tuned hypaer parameter in GroupKFold training and apply same parameters for the training with all data (model2, model3).
- Freezing Layers

**Did not worked**
- Text cleaning
- AWP
- Back translation augmentaion
