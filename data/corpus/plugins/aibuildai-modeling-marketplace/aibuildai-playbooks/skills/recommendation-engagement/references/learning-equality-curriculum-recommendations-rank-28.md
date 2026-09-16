# 28th place solution

Competition: learning-equality-curriculum-recommendations
Rank: #28
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394955

Thanks to the organizers for this interesting competition! I joined one month before the competition ended and am happy that a simple and direct approach got me to 28th place.

# Stage 1 retriever: Biencoder
Teach model to embed related topics and contents closely together, using the following common setup:
* Embed topic and content using same transformer backbone and mean pooling
* Compute cosine similarity between all examples in a batch
* Multiple in-batch negative ranking loss with temperature 0.05

The following modifications gave a significant boost to the model:
* Modify sampler such that all examples in a batch share the same language. This makes the negative examples much more meaningful. For inference, I only consider candidate contents with matching language.
* Compute cross-entropy loss row-wise and column-wise (with respect to topics and contents)
* Penalize scores for the correct class with a margin

The best retriever used XLM-RoBERTa (large) as backbone and was trained for 7 epochs. On my CV, it achieves a recall@50 of 91%.

# Stage 2: Crossencoder
For every topic generate 50 candidates using the biencoder. The crossencoder feeds a joint representation of topic + content through a transformer and does binary prediction whether they match or not. The stage 2 model was trained using the out-of-fold predictions of the stage 1 model as input.

The only things that worked for me in improving model performance were oversampling the positive class and using differential learning rates.

My best stage 2 model was a multilingual BERT trained for 12 epochs achieving 63.6% on my CV (64.3% and 68.3% on public and private LB respectively).

Looking at its performance and predictions, I realized that the crossencoder was having a hard time predicting a match from the text representations alone and experimented with GBDT as stage 2 model, but was not able to come up with a better-performing model.

# Input representation
Could not find a input representation that worked better than simple concatenation.
- Topic representation: Title. Parent title. ... Root title. Topic description.
- Content representation: Title. Description. Text.

Sequence length was 128 tokens for both models.

# Cross validation
I used 5 folds split on topics for stage 1, and 3 folds split on non-source topics for stage 2. I had good correlation between CV and LB using this setup. Splitting on channels instead led to unstable CV for me. Probing the public LB revealed that in fact there was a great amount (over 40%) of near-duplicate topics, so I settled in favor of a simple CV setup that gave me good correlation with LB.

My evaluation metrics for stage 1 were average precision and recall@N, and directly the competition metric for stage 2.

# Post processing
* Topics with no predictions get assigned the nearest neighbor according to biencoder
* If a new topic shared the same title, parent title, and grandparent title as a seen topic, add the contents of the seen topic to the predictions (tiny boost).
