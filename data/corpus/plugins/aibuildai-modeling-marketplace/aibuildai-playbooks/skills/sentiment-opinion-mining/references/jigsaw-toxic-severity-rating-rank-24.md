# Reranking with Late Interaction & More

Competition: jigsaw-toxic-severity-rating
Rank: #24
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306205

## Background

Have you ever scrolled through a 500 comment Reddit thread where every reply is worse than the last?

Did you close the app and move on?

Well I have done the exact opposite. COVID has given me way too much free time and I have spent my entire 4th semester of college learning the latest and the greatest NLP techniques to compute the exact hierarchy of unbearability of these types of Reddit comments.

The goal of this competition is to rank 14000 human-made comments based on their relative toxicity severity with the evaluation metric being **Average Agreement with Annotators** which means that models should accurately rank pairs of comments (less_toxic vs. more_toxic) as human expert annotaters did in the hidden test set.

I looked at the raw dataset and I have not read this many slurs since I played Clash of Clans in 10th grade. I would like to dedicate this solution writeup to the human annotaters who had to sit and label this dataset. 

GPT3 API had just came out during this competition and I decided that I am going to spend the next 6 months diving as deep as possible into the SOTA of NLP. I have been very active in this competition and have written a lot of discussion posts on my progress as I dive deeper into the NLP world. 

Since I am still too much of a noob to be able to publish a paper in ICLR or NeurIPS by myself I have been treating Kaggle discussion boards as my open source research journal. Some of my discussion posts may be a *little* out there but I would encourage you to give them a read as you would definitely learn something new even if you are an expert:

*   **[Why your RoBERTa model isn't working?](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/287677)**
*   **[Ok Boomer, Time to Start Using TPUs](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/286833)**
*   **[Let's inject Steroids into the AdamW Optimizer](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/288996)**
*   **[I am Having a Crisis of Faith (in Deep Learning)](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/295624)**


---

## Late Interaction Mechanism for Reranking

The idea is loosely taken from but is not exactly similar to the late interaction mechanism of the recent [ColBERT](https://arxiv.org/pdf/2004.12832.pdf) paper. 

The goal of the competition is to rank 14000 comments based on their level of toxicity. I thought this problem was somewhat similar to search ranking on Google where a model must rank millions of search results given a query. I started reading papers on search ranking which led me to being introduced to the field of Information Retrieval. This is where I came across Cross-encoders, Bi-encoders and the Late interaction mechanism.

I have made a post about [Cross Encoders vs Bi Encoders](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/296349) previously. Late interaction mechanisms can be seen as an intermediate step between Bi Encoders and cross encoders with the latency and accuracy falling between the two.

### Cross-Encoder Approach
[Kaggle Notebook: Train Cross Encoder with Aux Outputs](https://www.kaggle.com/code/readoc/jigsaw-train-cross-encoder-with-aux-outputs?scriptVersionId=313854961)

We are given the training data in pair form of (less_toxic_comment, more_toxic_comment). It's very easy to train a cross-encoder on this dataset as we can feed the inputs to the the model and attach a binary classifier on top.

* **Data Input Format:** `[CLS] Comment_1 [SEP] Comment_2 [SEP]`
* **Mechanism:** The backbone performs cross-attention between the two comments to determine whether Comment_1 is more toxic than Comment_2.
* **Output Head:** A binary classifier that predicts whether Comment_1 is more toxic than Comment_2.

Since BERT is already trained on sentence similarity tasks where the sentences are sepererated by the `[SEP]` token, we can use the same data format to train the model. A cross-encoder would be able to better model this pairwise interaction but it cannot be used in test time because there are total of 14k x 14k = 98M possible pair interactions between the comments. We simply do not have enough compute for 98 Million inferences.

### Bi-Encoder Approach
In the Bi-encoder approach, instead of feeding both comments to the model together, we process each comment independently through the same BERT-type model (a shared-weight Siamese network). This produces a single scalar toxicity score for each comment.

Now we have to train the model in a way so that for any pair the score for the `more_toxic` comment is higher than the score for the `less_toxic` comment. This is where we can use [MarginRankingLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html).

### Cross Cross Encoder Approach (In Context Ranking)
I had a very experimental approach in mind. I noticed that on average each comment takes 30-60 tokens with 64 tokens being 98th percentile. This means that realistically we can fit upto 8 comments in the context window of a 512 token sequence length model like BERT and RoBERTa.

So I thought instead of ranking pairwise why not rank everything at once? The GPT3 API just got released by OpenAI and it supports a 2048 token context window. We can fit ~32 comments in the model context and ask the model to rank 32 comments at once. This would not work for 2 reasons: 

1. We cannot infer with GPT3 API in the hidden test set as the code does not have internet access during inference time.
2. Even if we could use GPT3 API, I still cannot train GPT3 directly.

---

## Pseudo Labeling

I built an iterative Pseudo labeling pipeline following [this paper](https://arxiv.org/abs/1911.04252). Just using the 2018+2019 competition dataset (which contained 200k+ samples of toxic comments drawn from the same data distribution) gave me a big improvement in the score. 

Then I ported models from the highest-scoring public notebooks and experimental models like HateBERT to generate even more Pseudo labels. For each comment, I had 20+ pseudo labels. 

| Round | Public LB |
| :--- | :--- |
| Round 0 | 0.833 |
| Round 1 | 0.837 |
| Round 2 | 0.844 |
| Round 3 | 0.842 |

---

## Modeling and Loss Function

I tried the wackiest possible models and loss functions I could find including using HateBERT as the base model and very untraditional custom written loss functions.

* [Some Ideas for the Loss Function](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/299262)
* [The Best Way to Utilize 2016 Competition Data](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/295639)

In the end I ended up using the **Dice Loss** with alpha 0.25 from the paper [Dice Loss for Data Unbalanced NLP Tasks](https://arxiv.org/abs/1911.02855) to train on the binary labels: `severe_toxic`, `identity_hate`, `threat`, `toxic`, `obscene`, and `insult`. 

I got the best performance from the **Muppet RoBERTa Large** backbone by Facebook. Unlike RoBERTa, it was trained on sentiment and sentence similarity tasks which could be the reason for its better performance. I also trained a model on the [Ruddit](https://aclanthology.org/2021.acl-long.210/) dataset.

---

## TPU Optimization

I love love love TPUs. Thank you Google for giving me 20 hours of free 8xTPUv3 every week. I have tried building custom models specifically optimized for TPUs. Check out my [XFormers GitHub repo](https://github.com/sarthak-314/xformers). 

To speed up the training I used a **batch size of 512** with mixed precision. With this training a Roberta large model on 200k+ comments took <5 mins / epoch. I generally use Tensorflow+TPU for NLP because 20 hours of free 8xTPUv3 >> 30 hours of free P100 GPU. 

### Hyperparameters
To make up for the large batch size I set **beta_2 to 0.99** and used a cyclic learning rate with a max LR of 16e-5. Most people do not think about this beta_2 value in Adam Optimizer. Definitely check out my post on [Why your RoBERTa model isn't working?](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/287677).

---

## SWA, Moving Average and Model Souping

I also implemented a **SWA (Stochastic Weight Averaging)** callback in Keras to checkpoint the weights at the end of each cycle. I averaged the weights in the end (Model Soup).

* **Moving Average:** Small lift with using EMA of 0.99. 
* **Memory:** Doing EMA consumes double memory for storing 2 copies of model weights but because I got TPUs I don't care. 

### Ensembling
Ensembling on a single backbone did not help because of soft labels of pseudo labels. I found an interesting paper which said that Ensembling and Pseudo Labeling does not work very well together. 

---

## Hindsight

I really wish I had read this paper by Deepmind: [Multiplicative Interactions and Where to Find Them](https://openreview.net/pdf?id=rylnK6VtDH). 

I tried training an MLP on concatenated sentence embeddings vectors with the goal of differentiating less_toxic and more_toxic vector. If I had read the paper I would have known that multiplicative interactions (like vector dot product) would outperform this convoluted architecture.
