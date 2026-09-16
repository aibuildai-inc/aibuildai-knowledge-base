# 7th Place Solution - A good CV is all you need

Competition: isic-2024-challenge
Rank: #7
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532687

Hi all,

I would like to express my gratitude to the organizers for hosting this competition. On behalf of my team, I’d like to share our approaches:

#**Cross-validation (CV):**
We tried three different CV strategies: (1) GroupKFold on patient_id, (2) GroupKFold on patient_id and stratified on attribution (hospitals). However, these two CVs are sometimes not correlated with LB. Therefore we came up with the third and final CV (3) which simulated the scenario **where two new hospitals appear only in the test set** as in [the discussion](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/523024). Specifically, we **trained on data from five hospitals** and **predicted on seven hospitals**. The intuition to design this strategy is that we want to evaluate the performance of the model on both old and new hospitals, because the data and label may be different between hospitals (due to measuring errors, doctor bias, etc.). We found a **very high correlation** between the third CV and the public LB and used this CV until the end.

#**Solution:**
Our solution is the weighted ensemble of three tree-based models. In each tree-based model, we used the metadata and the prediction from CNN models as features. For CNN models, we trained multi-head models (1 for the classification target, 1 for predicting lesion_id) using variants of backbone.

**1. CNN models:**
- We used ResNet18, EfficientNetB0, EfficientNetB1, and Swin_Small as our backbones.
- During exploring features we noticed that all the positive samples had the lesion_id field (which is only available in training data), which means that samples with `lesion_id` will be more suspicious than other samples. So we leveraged this `lesion_id` information by adding another head to our CNN. This helped improve the pauc of CNN compared to using the classification target only.
- Pre-training on data from previous ISIC competitions.
- Experimented with various strategies, such as training with a weighted sampler, dual sampler, and using pauc loss, but only the weighted sampler proved effective.
- The results of combining image and tabular data were better than using image data alone. However, the inference process for these models was time-consuming, so we decided not to include them in the final submission.

**2. Tree-based models:**
- 3 variants of boosting tree: CatBoost, XGBoost and LGBM.
- Use public feature engineering from [the public kernel](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data).
- Group and aggregation information by `patient_id`, `tbp_lv_location` and `attribution`.
- Sample weight: 1 for negative, x for lesion and y for positive (the tuned value for x and y vary on different tree lib).
- Cross-feature (OOF prediction) from CNN models.

**3. Final submission:**
- The final submission was a weighted ensemble of three tree-based models and the neural network model on tabular data. 
- Because we 100% trust our CV, so 2 submissions we choose are all generalized well on both the CV and LB. Our submissions with **the highest score on CV** are also **the highest score on both public LB and private LB**.

**4. What did not work:**
- Multi-label for CNN models with the target and confidence score.
- Stacking is complicated and not as good as the weighted ensemble.
- SVM, TabNet, and other classifiers.
- Remove hair from skin images using OpenCV as in [the discussion](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/519735).
- Under-sampling as suggested on the public kernel.


We had a ton of work to get through each day and we also had daily meetings to go over what was done and figure out what needed to be done next, which really helped keep us on track and moving forward.

This competition has been such a great journey, and I’ve learned so much from it. I want to extend my sincere thanks to Mr. Duc @mathormad , Mr. Tu @minhtu123 , and Ms. Linh @linhlethuy for teaching and tirelessly helping me in fixing bugs for many hours or even days. I truly appreciate it. 

I would also like to thank Kagglers for their valuable discussions and insights.

HARD WORK PAYS OFF!
