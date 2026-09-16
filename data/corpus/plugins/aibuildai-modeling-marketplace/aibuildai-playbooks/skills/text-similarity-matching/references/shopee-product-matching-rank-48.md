# 48th Place Silver - Simple Baseline

Competition: shopee-product-matching
Rank: #48
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238263

Congrats to all participants! I am really glad to obtain my first silver medal and excited to become a competition expert :)

There is nothing fancy with my solution. After reading the other shared solutions, i think the only thing worth sharing is my model training methods (which is what differentiated me from public kernels).

# Summary:
b6 and bert-base-multilingual, both with arc face layer. I followed the BN-512-BN architecture as described in the arcface paper. 

I then normalized each embedding, concat them, and applied KNN using cosine metric to all 3 embeddings and took the union as matches.

My 'post-processing trick' was to combine similar matches (measured with Jaccard index) using majority voting. This trick, however, only boosted LB by +0.001.

| Method | Public  | Private |
| -- |  -- |
|  bert base multilingual uncased + TTA | 0.636 | 0629 |
| b6 effnet (noisy student) | 0.696 | 0.688 |
| b6 + bert + concat | 0.750 | 0.739 |
| with 'post processing' | 0.751 | 0.740 | 


### Validation Scheme
Group KFold with 2 splits then use KNN to get CV score. I then retrained the models using the full train data and made a submission to get my public LB.

I only accepted changes if there is an improvement to both CV and public LB.

### Model Training
First, I choose the backbone which gave me the best CV/LB:
- Bert base multilingual uncased (out of XLM-Roberta and bert base)
- B6 Noisy Student Version (out of B0-B6 AA/RA/NS variants)

1. "Pretraining" -> I first froze the backbone layer and then trained the embeddings/arc face layer for a few epochs before "finetuning" by unfreezing all layers and then training for a few more epochs.
2. Freezing all the batch norm layers for b6 backbone
3. GeM pooling layer worked for b6 while pooling did not help bert.
4. Image augmentation adopted from [here](https://www.kaggle.com/dimitreoliveira/flower-with-tpus-advanced-augmentations?scriptVersionId=0)
5. label smoothing helped, but impact was minimal

### Inference
- Snapshot ensemble (3x for bert and 2x for b6)
- TTA for text inputs. I generated embeddings for original titles and a clean variant (unidecode, remove English/Indonesian stopwords, etc.) 

I ensembled each of them separately by taking the average, before normalizing them.

I simply concatenated the text (512) and image embeddings (512) to get a 'text + image embedding' (1024).

I applied KNN on all 3 embeddings separately and then took the union to get the ensembled matches.

### 'Post Processing Trick'
For each of the predicted matches, I tried to find similar predicted matches and merge them together to get my final matches.
- Using Jaccard Index as a similarity score, I collected all matches with Jaccard Index >= threshold then I kept only 'posting ids' which appeared >= 2.
- I used different thresholds depending on the length of the ensembled matches.
