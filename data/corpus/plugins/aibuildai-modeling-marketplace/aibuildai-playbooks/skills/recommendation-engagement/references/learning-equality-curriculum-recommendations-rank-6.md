# 6th place solution

Competition: learning-equality-curriculum-recommendations
Rank: #6
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394813

# Summary
* Retriever and reranker
* Transformer-based model for retriever
* GBDT-based model for reranker
* Inference code https://www.kaggle.com/code/iiyamaiiyama/llecr-ens8910-125128-149154-155157





# CV strategy
Simple 5fold random split with topic-id.
For stage 1 I used all data, including "source".
For stage 2, I used the same fold split as for stage 1, and created several models with and without "source" to ensure diversity.
CV and LB were well correlated.

I trained a model based on channel GroupKFold and select it for one of my final submissions, but a simple random kfold gave slightly better results for CV and LB.


# Retriever(stage1)
## Text
The training text was created as follows:
* Topics  
  Recursively traversed to the root node and added titles. Finally, the topic description was added. For example, (root title + parent1 title + ... + topic title + topic description).  
* Contents  
  The title and description were concatenated. The "text" column was discarded.

model	max positive score@50	CV F2@stage1	CV F2@stage2	public LB@stage2
sentence-transformers/LaBSE	0.8887	0.5462	0.6727	0.676
sentence-transformers/paraphrase-multilingual-mpnet-base-v2	0.8891	0.5429	0.6698	0.678
facebook/xlm-v-base	0.8869	0.532	0.669	0.671
xlm-roberta-base	0.8832	0.5388	0.6666	0.676
naive ensemble above four	0.9336	-	0.6916	(I didn't sub this)
my published submission 	-	-	0.7152	0.707


## Model
Each topic and its correlated contents were grouped together as one class. 
The model was trained with ArcFace. Each model produces 768-dimensional embeddings. 
Trained for 30 or 60 epochs, which took about 5 hours per fold. 
The margin was gradually increased from 0.2 to 0.6 during training.
The following models were used for the final submission.

* [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)
* [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
* [facebook/xlm-v-base](https://huggingface.co/facebook/xlm-v-base)
* [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)

## KNN
For each topic, find 50 nearest neighbor contents and these pairs are passed to the reranker(stage2).

# Reranker(stage2)
When I submitted stage1 model only, I coudn't reach 0.6 on the public LB. I think this is because stage1 model doesn't have any topic-tree structure information. So I added many tree-based features for stage2 model.

## Model
CatBoost and XGBoost were used for reranker.
Features were created for each pair, then GBDT model predicts probability that the pairs were correlated or not.

## Example of features
* Embeddings distance
* TF-IDF distance
* Whether the topic's siblings correlated the target content
* How many times the content was correlated in the channel

## Ensemble
* stage1  
  Concatenate embeddings then find KNN
* stage2  
  Average GDBT predictions, then select pairs as positive predictions above the threshold.

# Post-processing
* After stage2, if there were no predictions for a topic, the content with the highest predicted score was added.
* If two contents that were always correrated together in "correlations.csv", and one of them appeared in the prediciton, add the other one.
* Search best threshold for each channel
  For channels not included in the training data, a fixed threshold was used. (based on CV across all channels)

# Not worked
* "text" information of contents  
  I cannot find good way to use "text" column.
* Transformer-based reranker  
  They were very prone to overfitting. 

# Appendix: my models
|Model|max positive score@50|CV F2@stage1|CV F2@stage2|public LB@stage2|
|:----|:----|:----|:----|:----|
|sentence-transformers/LaBSE|0.8887|0.5462|0.6727|0.676|
|sentence-transformers/paraphrase-multilingual-mpnet-base-v2|0.8891|0.5429|0.6698|0.678|
|facebook/xlm-v-base|0.8869|0.5320|0.6690|0.671|
|xlm-roberta-base|0.8832|0.5388|0.6666|0.676|
|Naive ensemble above four|0.9336|-|0.6916|(I didn't sub this)|
|My published submission|-|-|0.7152|0.707|
