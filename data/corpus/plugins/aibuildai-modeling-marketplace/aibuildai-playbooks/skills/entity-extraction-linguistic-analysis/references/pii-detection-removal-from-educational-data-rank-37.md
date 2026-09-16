# 37th place solution: Deberta voting ensemble

Competition: pii-detection-removal-from-educational-data
Rank: #37
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497293

Firstly huge thanks to the organizers for hosting this competition. Also, thanks to everyone for the insightful discussions, data and code shared. I'm glad to have made it to my first silver medal!😄

**Approach**

My solution is essentially an ensemble of deberta-v3-large models trained on different data subsets. 

As pointed out [in this post](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/473011)  regarding the presence of PII data at the end of the essay, I used **return_overflowing_tokens**, **striding** in the tokenizer and **wids** to map each token to the corresponding text in the essay tokens array. In this way, there was no need to worry about a large sequence length. I used max_length=512 and stride=128.

My hold out validation set was the training data's fold%4==2. I chose it because it has majority of the class labels and also showed good correlation with the LB. My training subsets were a mix of given training data and external datasets such as - 
1. [Nicholas data](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221) + train data folds%4=(0,1,3)
2. [Mpware data](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/477989) + train data folds%4=(0,1,3)
3. [1850 mistral](https://www.kaggle.com/datasets/tonyarobertson/mixtral-original-prompt) + train data folds%4=(0,1,3)
4. [2k mistral](https://www.kaggle.com/datasets/mandrilator/pii-mistral-2k-fit-competition-v2) + train data folds%4=(0,1,3)

I followed [this post](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/470978) and found the best threshold for each model based on the validation set performance. After thresholding, I ensembled by voting among the deberta models. 

**Code:** [Train code](https://github.com/rush2406/pii-detection/blob/main/train.py), [Inference code](https://github.com/rush2406/pii-detection/blob/main/inference.py)

**Things which didn't work**

Training on custom generated data similar to [this](https://www.kaggle.com/code/minhsienweng/create-ai-generated-essays-using-llm). Maybe a better prompt would have worked 😅
Training/ using Longformer, LLM predictions
Weighted average ensembling
Using class weights in loss functions to handle class imbalance
