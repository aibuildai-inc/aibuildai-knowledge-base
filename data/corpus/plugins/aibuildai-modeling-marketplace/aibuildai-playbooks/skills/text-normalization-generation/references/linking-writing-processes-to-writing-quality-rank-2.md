# [3rd place solution] Trust CV is all you need.

Competition: linking-writing-processes-to-writing-quality
Rank: #2
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466775

I want to express my gratitude to the host for organizing this great competition, and I feel incredibly fortunate to have experienced such a significant shake-up!!!!!!😄
I chose to trust my cv, which is about 0.580, and as I expected, the shake-up happened. While I anticipated some shake-up, I never expected to climb to the 3rd place.

The key component of my method is to **align context features generated using a pre-trained Deberta-based regressor**. 

Firstly, using the reconstructed essays [e.g., Essays reconstructed](https://www.kaggle.com/datasets/hiarsl/writing-quality-challenge-constructed-essays), I trained a Deberta regressor by adding 3 nn layers at the bottom of Deberta. This procedure follows the NLP process like [Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data), except that the characters are replaced by "anonymous" character q. Please find the [code](https://www.kaggle.com/code/leonshangguan/training-deberta-based-regressor-as-fe-extractor?scriptVersionId=158377154) for more details. Training the Deberta-based regressor can achieve a cv about 0.75.

After that, I dropped the last layer of the Deberta regressor, and used the second last layer's output, which is a 128-dimensional vector as the **context feature** extracted by the language model (i.e., Deberta).

Based on the features from public notebooks that performed well on the public leaderboard (mainly from [Writing Quality(fusion_notebook)](https://www.kaggle.com/code/yunsuxiaozi/writing-quality-fusion-notebook) by @yunsuxiaozi), I trained lightgbm/xgboost/catboost and save the feature importance, then selected top 64/128/256 most important features from each model and take their union.

Finally, the selected features from the public notebook and my context features generated using Deberta were concatenated. I trained [lightgbm/xgboost/catboost models](https://www.kaggle.com/code/leonshangguan/all-fe-of-tree-models-training-pipeline-saved-fe?scriptVersionId=158376408) which give cv around 0.588 to 0.595, and [nn models (mlp/autoint/denselight)](https://www.kaggle.com/code/leonshangguan/all-fe-of-nn-models-training-pipeline-saved-fe?scriptVersionId=158376311) which have cv around 0.580-0.590. For the nn models, thanks to [this notebook](https://www.kaggle.com/code/alexryzhkov/lightautoml-nn-test) by @alexryzhkov .

Due to the time limit, I only trained a Deberta-base model as the context extractor, but I believe other language models (e.g., Bert) and even llms (e.g., llama) are worth trying as well. As Deberta and nn models require GPU resources, I didn't consider the efficient track. 

**All code are listed below:**
Feature generation and processing: [GPU all in training with context fe generation](https://www.kaggle.com/code/leonshangguan/gpu-all-in-training-with-context-fe-generation/notebook) The notebook shows feature engineering and tree-based model training. For convenience, I saved the processed features to a .csv and train nn/tree models separately.


Training: For each model, I train with 10 to 15 different random seeds and 5 to 10 folds (StratifiedKFold). 
Deberta-based regressor: [Training Deberta-based regressor as fe extractor](https://www.kaggle.com/code/leonshangguan/training-deberta-based-regressor-as-fe-extractor?scriptVersionId=158377154)
nn models: [All fe of nn models training pipeline saved fe](https://www.kaggle.com/code/leonshangguan/all-fe-of-nn-models-training-pipeline-saved-fe?scriptVersionId=158376311)
tree models: [All fe of tree models training pipeline saved fe](https://www.kaggle.com/code/leonshangguan/all-fe-of-tree-models-training-pipeline-saved-fe?scriptVersionId=158376408)

Inference: [GPU all fe inference](https://www.kaggle.com/code/leonshangguan/gpu-all-fe-inference?scriptVersionId=158374031) I didn't tune the weights for each model, just some random numbers based on cv. A simple postprocessing `scores[scores<0.5] = 0.5
scores[scores>6.0] = 6.0` was conducted before submission.


**Thank you all and feel free to comment if you have any questions**
