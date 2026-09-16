# 5th Place Solution

Competition: shopee-product-matching
Rank: #5
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238078

Congrats to all medal winners and thanks to my teammates @tereka @allvor for this nice teaming experience. I will share our team's solution. We basically had 2 kernels, which are very similar but not the same. They also scored almost the same but due to time limitations, we couldn't combine them. I will first explain the kernel that I mostly worked on ( https://www.kaggle.com/aerdem4/shopee-v03 ), then I will state the difference with @tereka's kernel.

**CV Scheme**
We have split the data into 2 folds. First fold is used for training representation learning models (arcface), second fold is used for evaluating them. We have also trained 2nd stage models on the second fold with GroupKFold.

**Representation Learning**
From the beginning, I have tried to train a model that can generate vector representations for images and texts in the same space. I thought this approach can have good regularization effect. I managed to do it but there was no significant improvement for our validation score. Therefore we ignored this idea and focused on having separate models for images and titles.

English Distilbert and Indonesian Distilbert models are trained for obtaining the title vectors. For the image vectors, ViT and Swin Transformers are trained on size 384 with relatively heavy augmentations and also EffNet B4 on size 512. We ensemble these models with vector concatenation.

**Weighted Database Augmentation**
We have matched the vectors and get the closest match for each posting id. We updated the vector representations as weighted mean of themselves and their closest matches. Weights are based on cosine distances of the matches. If a match is far for a vector, then that vector stays unchanged. I believe this approach works really well for regularization of the vectors utilizing the information that each posting has at least one match.

**Matching**
We used cuML's NearestNeighbors for matching the vectors and used a threshold for filtering.

**2nd Stage Model**
We have extracted some features for matched unique pairs and fed them to XGB model which runs on GPU. The features are "img_dist", "text_dist", "dist", "dist_rank", "cos_sim", "cos_sim2". Basically, vector distances, their ranking within each posting id, 2 different tfidf cosine similarity with different parameters.

Since test set size was larger than our one fold size, some features were going to have different distribution on the test set due to higher possibility of False Positive matches. Therefore another XGB model is trained on percentage rank features and ensembled with the previous.

**FP Features**
Since test set and train set have no match, one can use training set for determining how easy it is for a posting to have FPs. We used closest match distances from training set as features. This method improved our CV score significantly but improved LB relatively less for me and no improvement for @tereka's kernel. I believe this is again due to the size difference between the train and test.

**Agglomerative Clustering**
Once we have match probability predictions from XGB models, we can use a threshold and match the postings. But we did something a bit more complex. We sorted all pairs by their match probabilities and started matching from the most likely match. Each match above 0.8 probability, merges their clusters. Each posting with no cluster can be matched to a cluster if the probability is above 0.7. Each posting with no cluster can be matched with each other if the probability is above 0.3. This method allows to have confident clusters and not-so-confident many pairs.

Besides above, @tereka had some different image models and MLP model for title char2grams. There is a video explaining our solution in Japanese: https://youtu.be/vtI8P-ttPrk?t=1915
