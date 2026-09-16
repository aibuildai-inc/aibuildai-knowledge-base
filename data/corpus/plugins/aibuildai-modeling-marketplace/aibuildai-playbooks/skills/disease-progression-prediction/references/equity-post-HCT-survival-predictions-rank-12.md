# 12th place solution

Competition: equity-post-HCT-survival-predictions
Rank: #12
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566584

Given the similarities between this competition and the Tabular Playground Series competitions, and the fact that both my teammate and I are experienced TPS participants with a history of success, we decided to apply our knowledge of synthetic tabular datasets to this competition.  

**TL;DR: Large ensembles with minimal feature engineering, plus a few additional tricks.**  

Over the course of three months, we trained a large repository of models and experimented with various ensembling techniques. By the end of the competition, we had accumulated a total of 98 models trained on a variety of different target transformations. Given the large number of base models, we also did feature selection during ensembling to reduce the number of models and improve our CV score.  

## Final Solution  

Our final approach consisted of a **multi-layered ensemble**:  

1. **First Layer:** A diverse set of base models, including gradient boosted tree models, neural networks, and AutoML.
2. **Second Layer:** Three different ensemble methods—Ridge regression, hill climbing, and AutoGluon. We conducted feature selection separately for each of these ensemble methods.  
3. **Third Layer:** A weighted ensemble of the OOF predictions from the three second-layer models.  
4. **Final Step:** We used an LGBM classifier and applied the masking trick to obtain our final predictions.  

Below, we provide the CV scores for all our base models and ensemble models.  


## Acknowledgments  

We would like to thank the organizers for an interesting competition, as well as everyone who shared their insights and ideas, especially @ambrosm, @adaubas, @andreasbis, and @mtinti.

On a personal note, I want to congratulate, and give a big thank you to my teammate, @rzatemizel. This was our first competition medal, and I couldn't have asked for a better partner!
