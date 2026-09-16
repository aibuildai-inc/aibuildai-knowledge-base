# [Public 21th Private 7th Solution] 14 models in Ensemble and postprocessing

Competition: pii-detection-removal-from-educational-data
Rank: #7
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497310

**Link to the notebook:** https://www.kaggle.com/code/mikhailgolubchik/fork-of-the-lalab-pii-data

First of all, a big thank you to those who trained excellent models that we used in this notebook. Those who created and published their notebooks and shared comments and insights.

During Inference, we used 14 models from the following public datasets:

https://www.kaggle.com/datasets/verracodeguacas/pii-deberta-models
https://www.kaggle.com/datasets/emiz6413/37vp4pjt
https://www.kaggle.com/datasets/startalks/pii-models

Special thanks to @emiz6413 for his superb notebook and model. With a description of the training process.

**Ensemble**

A large number of models were used in the ensemble, as for the first three hours of the submit, the best models made predictions on the entire dataset. And for the remaining five and a half hours, the remaining models additionally made predictions on 2/3 of the dataset with the shortest token length.

We assumed that the number of PII tokens does not depend so much on the length of the text. And the prediction time depends on the length of the text. Therefore, it would be more profitable to use a larger number of models on shorter texts. And to predict longer texts with a smaller number of models.

In addition, different thresholds were set for different types of labels. For student names, the probability threshold was lower. For other labels, the threshold was higher. Since there were fewer other labels in the training dataset, and we assumed that the model learned worse how to find them and at a lower threshold would predict many unnecessary labels, than for student names.

**Post-processing**

- Student names should start with a capital letter and continue only with lowercase letters:
r'^[A-Z][a-z]+$'
- For "B-" tokens not followed by "I-" tokens, we removed tokens that were too short, as well as phone numbers and email addresses, and B-ID_NUM tokens that did not match the pattern. For example, for B-ID_NUM, there should have been at least two consecutive digits in the token, and it should have been at least 4 characters long. "B-" tokens followed by "I-" were not cleaned with these additional patterns.
- PII labels were added for tokens with addresses if line breaks and other tokens were not marked inside the address.

**Conclusion**
Thank to my teammate [@wasjaip](https://www.kaggle.com/wasjaip). It was an interesting time and a good study.
