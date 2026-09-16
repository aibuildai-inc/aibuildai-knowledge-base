# 1st Place Solution - From Embeddings to Matches

Competition: shopee-product-matching
Rank: #1
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238136

Congratulations to all the winners, and thanks to the hosts for arranging this interesting dataset and Kaggle for arranging this competition! Also, great thanks to my teammate @limerobot who always comes up with brilliant ideas and never gives up. I learn a ton from him.

Title is 'From Embeddings to Matches', since we experienced major CV/LB improvements from *properly utilizing image/text embeddings to search matches*. On the other hand, individual model improvement helped less, that is, 'from input to embedding' mattered less after we've got decent models. Even when CV for each image/text model increased quite a lot, CV for ensemble and LB didn't increase much.

I'll first share our history of public leaderboard scores. I'll only write things that improved our score significantly. Note that model/training optimization improved the score here and there, too.

* baseline: **0.7**(image only), **0.64**(text only)
* concat img_emb & txt_emb -> normalize: **0.724**
* *min2*: **0.743**
* normalize -> concat img_emb & txt_emb: **0.753**
* full data training: **0.757**
* *union comb, img, txt matches* & tune threshold: **0.776**
* *INB* & add diverse txt models: **0.784**
* *use img, txt, comb emb at INB stage 1* & tune threshold jointly: **0.793**

Now I'll go into details of our solution.



## 1. Model

We used two `eca_nfnet_l1`s from `timm` for image encoders, and `xlm-roberta-large`,`xlm-roberta-base`, `cahya/bert-base-indonesian-1.5G`, `indobenchmark/indobert-large-p1`, `bert-base-multilingual-uncased` from `huggingface` for text encoders.

We used `ArcFace` to train the model. After pooling from image/text encoder, we applied batchnormalization and feature-wise normalization to output embedding. Following the `ArcFace` paper, the normalized embedding is multiplied with normalized weight matrix to produce cosine, then we apply arc margin and softmax.

[model]


## 2. Tuning ArcFace

It was the first time for both of us to use ArcFace, so at first we had a hard time tuning it. Sufficiently large margin was important for the quality of the embedding, but when we increased it, the model suffered from convergence issue. We found several ways to overcome this issue, and finally applied among these options. 

* increase margin gradually while training
* use large warmup steps
* use larger learning rate for cosinehead
* use gradient clipping

Margin of 0.8~1.0 was optimal for image model, and 0.6~0.8 for text model. When using the increase-margin approach, we start from a margin of 0.2 and increase it to 1.0 for image model and 0.8 for text model.

Also, we tried class-size-adaptive margin, which was introduced in the google landmark recognition competition solution (https://arxiv.org/pdf/2010.05350.pdf). It is best when margin is set to `class_size^-0.1` for image model and `class_size^-0.2` for text model. However, the improvement was subtle since the class was less imbalanced than in google landmark recognition competition.

Using larger learning rate for cosinehead and using gradient clipping not only made the model converge, but also improved CV score a bit.

Adding extra fc layers after global average pooling hurts the model performance, while adding batchnorm before feature-wise normalization improved the score.



## 3. Combining Image & Text Matches- concat & union

We have image and text input in this competition, and it is crucial to utilize and combine these modalities well. These are the approaches that we tried.

* Produce text matches from text embedding, image matches from image embedding, then union text matches & image matches
* Concatenate text embedding & image embedding -> produce comb matches
* **Union comb matches & text matches & image matches**

Second method was much better than the first method and the last method was much better than the second method.

We normalize image embedding and text embedding then concat them to calculate comb-similarities. Actually, it is the same as calculating similarity from image embedding and another similarity from text embedding, then averaging them. Thus, we can *interpret* the last method as image below.

[combining image & text]

So we are accepting items that image embedding strongly suggests, items that text embedding strongly suggests, and items that image & text embedding both moderately suggests.

We also tried to jointly train image encoder and text encoder. However, multimodal model was always inferior to training separately, even when we initialized it with trained encoders.



## 4. Iterative Neighborhood Blending (INB)

Apart from combining image and text matches, it was also crucial to properly utilize embeddings to produce matches. We made a nontrivial pipeline for searching matches from embeddings.

Based on QE(Query Expansion) and DBA(DataBase-side feature Augmentation), we created a pipeline called **INB(Iterative Neighborhood Blending)**. Most of the ideas are shared with QE and DBA, but some details are different. INB pipeline consists of these components.

K Nearest Neighbor Search

We use faiss(https://github.com/facebookresearch/faiss) for knn search, and set k=51 (maximum 50 non-self matches + 1 self). We used inner product as similarity metric (the embedding is normalized so it is equivalent to cosine similarity)

Thresholding

We converted cosine similarity to cosine distance (`= 1-cosine similarity`) for some convenience in implementation, and obtained `(matches, distances)` pair that satisfies `distance < threshold`. For each item x, we call this `(matches, distances)` pair as "neighborhood of x".

**Min2**

When doing thresholding, we can ensure that there are at least 2 matches per query, since it is guaranteed by the competition description. We reject the second closest match *only if* the distance is over the min2-threshold, which we set high.

**Neighborhood Blending**

The intuition for neighborhood blending is straightforward. After knn and thresholding with min2, we obtain the `(matches, similarities)` pair for each item, we have a graph, where each node is an item, and the edge weight is the similarity between two nodes. Only the neighborhoods are connected. That is, nodes that didn't pass the threshold condition and min2 condition from the query node, are disconnected.

We want to use the neighborhood items' information to refine the query item's embedding and make the cluster clearer. In order to do that, we simply weighted-sum the neighborhood embeddings with similarity as weights and add it to the query embedding. So we **blend neighborhood** embeddings. We call it **NB(Neighborhood Blending)**.

[NB]

Above image illustrates how one step of neighborhood blending is performed on a toy example. Let's look at node A. Its embedding is `[-0.588, 0.784, 0.196]` and its similarity to node B, C, D is 0.94, 0.93, 0.52 respectively. Red line means two nodes are neighbors, so they are connected. Dashed line means two nodes didn't pass the threshold, so are disconnected. We apply neighborhood blending, then: 
[nb equation]
after applying this, hopefully like in the right graph in the image, we get more clustered, refined embeddings.

We can apply NB iteratively. After blending neighborhood for stage1 embeddings, we do knn search & 'thresholding with min2' to get stage2 `(matches, similarities)`. We apply NB again, to further refine the embeddings. We can iterate until the evaluation metric stops improving. This is where **Iterative** comes from.

Code is quite simple. (you need to have your `neighborhood_search` function implemented)

```python
def blend_neighborhood(emb, match_index_lst, similarities_lst):
	new_emb = emb.copy()
    for i in range(emb.shape[0]):
        cur_emb = emb[match_index_lst[i]]
        weights = np.expand_dims(similarities_lst[i], 1)
        new_emb[i] = (cur_emb * weights).sum(axis=0)
    new_emb = normalize(new_emb, axis=1)
    return new_emb

def iterative_neighborhood_blending(emb, threshes):
    for thresh in threshes:
        match_index_lst, similarities_lst = neighborhood_search(emb, thresh)
        emb = blend_neighborhood(emb, match_index_lst, similarities_lst)
    return match_index_lst
```

---

Full INB pipeline is summarized in the below image.
[INB]

We apply 3stage INB.
To get to the details, we first find `(matches, similarities)` pair from image embedding, text embedding and comb embedding, respectively. Then we union the matches. (for similarities, we use comb similarities, if not exists, image similarities, if not exists, text similarities)
Using joined `(matches, similarities)` and comb embedding, we do neighborhood blending to obtain stage2 embedding. Then we find `(matches, similarities)` pair and apply NB to obtain stage3 embedding. We get stage3 matches from this embedding.
We union it with matches found solely by image embedding, and matches found solely by text embedding to get final matches, which we submit.

**About Threhsold Tuning**
Note that we have 10 different threshholds to tune. 3 for stage1 txt, img, comb thresholding, 1 for stage2 thresholding, 1 for stage3 thresholding, 2 for stage1 img, txt thresholding that directly connects to final union, 3 for stage1~3 min2 thresholding. But after we've tuned other thresholds to reasonable range, major score boost came from **stage2 and stage3 threshold**. (we got good results when fixing stage1 comb threshold to 0 - just use min2 from comb) So **at last we only had to tune those 2 numbers** against LB.

#### Visualizations of Embeddings before/after INB
*2021/05/13 editted*
We made a notebook that visualizes the effect of INB on embeddings.
https://www.kaggle.com/harangdev/shopee-embedding-visualizations-before-after-inb
Here are some examples



## 5. ETC

* for image model, cutmix with probability 0.1 helped
* for image augmentation, using only horizontal flip was better
* madgrad optimizer (https://github.com/facebookresearch/madgrad) performed better or similar compared to Adam and SGD
* Using full data for training improved performance
* Limerobot dug into the triplet loss but he never beat the arcface loss 😿
