# 10th place solution...my quick write-up!

Competition: feedback-prize-english-language-learning
Rank: #10
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369373

**Acknowledgement**

Phew, the shakeup was massive! :) I want to thank the organizers, the Kaggle community, and the Kaggle team for this competition. Competitions like this are great opportunities for me to learn new things and update my machine-learning knowledge.

I also want to thank my teammates, @runningz, @kunihikofurugori, and @habedi, for their efforts and cooperation in this competition. 

**Quick Summary**

Our final submission was a blend of two submissions files:
- The first one was the model trained by @runningz, which was an ensemble of three deberta-v3-base models trained (his models were trained in PyTorch, and he used pseudo-labeling to improve the results). His models were based on his knowledge of the previous feedback competitions.
- The second one was the stacked ensemble of 22 models from the rest of the team. The information about the stacking process is as follows:
     - We used `iterativestratification` with the number of folds set to four and the seed set to 42.
     - A set of oofs based on
        - a model trained with deberta-v3-small, 
        - two models trained with deberta-v3-base, 
        - a model trained with deberta-v3-xsmall ,
        - a model trained with distilroberta-base,
        - a model trained with opt-125m,
        - a model trained with roberta-base,
        - two models trained with opt-350m,
        - a model trained with electra-base-discriminator.
     - We used a couple of transformers without training, just embedded the text, and trained simple regressors on the numerical vectors. These models where
        - a model based on deberta-v3-base embeddings along with a BayesianRidge regressor,
        - a model based on deberta-base-mnli embeddings along with a BayesianRidge regressor,
        - a model based on deberta-v3-large embeddings along with a BayesianRidge regressor,
        - seven models based on deberta-v3-large embeddings along with a KNN regressor with different values of k (k=2, 4, 8, 16, 32, 64, 128),
        - a model based on deberta-base-mnli embeddings along with a BayesianRidge regressor,
       - a model based on google-electra-base-discriminator embeddings along with a BayesianRidge regressor,
     - Finally, we trained a Lasso regressor on the oofs. 
     - The CV of the stacked model was 0.4473, and LB was 0.43.
     - In the case of the models trained based on different transformers, the CV ranged in from 0.45+ to 0.49+, with deberta-v3-base being the best model in terms of CV and LB
     - In the case of the text embedding models, the CV of our oofs (even the KNN models) was between 0.45+ to 0.54+, and the best CV belonged to deberta-v3-large with CV equal to 0.4518 and LB of 0.44 (it is fascinating to see that with just embedding and no training one can get to 0.44!)
     - We tried different regression models as the stacker, and the best in terms of CV was Lasso
     - Performing log transformation on the oofs and running the RobustScaler from SKlearn improved the CV a bit
