# 11th place solution

Competition: feedback-prize-2021
Rank: #11
Source: https://www.kaggle.com/c/feedback-prize-2021/discussion/313184

First of all, thanks to competition organizers for hosting this competition.
Thanks also to the community for sharing many ideas in Notebook and Discussion.

## Summary
We ensembled 6 models and did post process. Next, we extracted candidate predictionstrings and predicted whether each predictionstring was TP or not using XGBoost, etc. Finally, we set a threshold for each class and cut off the right edge of the predictionstrings.

## Models
We referred to the following notebook.</br>
https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-633
</br>
The following 6models were trained with different max_length depending on the model.
- longformer-large-4096 (max_length:2048)
- roberta-large (max_length:512)
- bart-large (max_length:512)
- funnel-transformer-large (max_length:512)
- distilbart-cnn-12-6 (max_length:512)
- deberta-large (max_length:1024)

## Ensemble
Since different models use different tokenizers, once the prediction results were converted to the character level, they were reaggregated into longformer token units.
Ensemble with different weights for each of the following positions.
1. 0 ~ 512
2. 0 ~ Funnel-transformer-large-token-len
3. Funnel-transformer-large-token-len ~ 1023
4. 1023 ~ </br>
Funnel-transformer-large-token-len is the length of the 512 token of funnel-large when it is converted back to longformer tokenizer units. It was about 720.

## Post Process
- Multiply each prediction by the coefficient
- Ensemble with predictions of neighboring token
- Extracted candidates for predictionstrings with reference to the following notebook. These candidates were not only for the class with the highest prediction, but also for the second highest class. The some predictionstrings were extracted where the start was not B-token.</br>
https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-633
- The following 3 models predicted whether predictionstrings were TP or not. The features were created by aggregating the ensemble results for each predictstrings. A threshold was set for each class, and only predictstrings with predictions above the threshold were left.
    1. XGBoost
    2. MLP
    3. LSTM
- Finally, predictstrings length threshold and percentage of clipping were set for each class, and The right end of predictstrings that exceeded the threshold was deleted by the set percentage.

## Edit
Inference notebook
https://www.kaggle.com/code/columbia2131/exp-038-ensemble-xgb-mlp-lstm-fe-fix-tail
training code
https://github.com/TakoiHirokazu/Feedback-Prize-Evaluating-Student-Writing
post process code
https://colab.research.google.com/drive/1J2NTaRSKi-X5SBXxvQtVJxokZIW0oq4L?usp=sharing
https://www.kaggle.com/code/columbia2131/tuning-exp038-right-edge-removing/notebook
