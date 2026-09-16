# 7th Place Solution

Competition: otto-recommender-system
Rank: #7
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383769

First of all, thank you for hosting this super exciting competition! And thank you to everyone for sharing many important insights in this competition. Discussions also have been very beneficial to us in our efforts to achieve this grade.

## 1. Overview
 
Our team consists of two members, Jack and toshi_k. Although both of us are competitions grandmaster, we have different strengths. Before making up a team, our approaches were totally different. Ensemble of two approaches cancelled out each weakness and boosted our team to the gold medal.

Our solution is composed of LightGBM part, GNN (Graph Neural Network) part and ensemble. Jack was in charge of LightGBM part. He trained the best solo model in our team. toshi_k was in charge of GNN part and ensemble. He trained unique models by modern deep learning.  It contributed +0.003 to the team score by ensemble.

The details of LightGBM part is described in section 2. GNN part is described in section 3. Ensemble method and result are described in section 4.



## 2. LightGBM part (by Jack)

### 2.1 Candidate Generation

Before I go into details, let me just say that the method described here is my own, but when I optimized Recall@20, there was little difference of performance between it and the [Chris's Public Notebook](https://www.kaggle.com/code/cdeotte/compute-validation-score-cv-565) on validation. Thus, it is unclear how much of an advantage it may have.

The basic idea is to approximate a kind of posterior probability (let's call it a candidate score) of aids based on the co-visitation matrix, and the candidates were selected from the top ones. These candidates are common to all three action types.

The last 10 aids and action types of the session were used to calculate the candidate score. That is, the score can be calculated as follows:

$$Score(aid) = P(aid | (aid_1, type_1), (aid_2, type_2), ..., (aid_{10}, type_{10}))$$

Here, we forcibly assume independence and conditional independence of each action, similar in concept to Naive Bayes,

$$\begin{eqnarray}
Score(aid)
&=& P(aid) \cdot \frac{P((aid_1, type_1), ..., (aid_{10}, type_{10}) | aid)}{P((aid_1, type_1), ..., (aid_{10}, type_{10}))} \\\\
&\approx& P(aid) \cdot \prod_{i=1}^{10} \frac{P((aid_i, type_i) | aid)}{P(aid_i, type_i)} \\\\
&=& P(aid) \cdot \prod_{i=1}^{10} \frac{P((aid_i, type_i), aid)}{P(aid_i, type_i) P(aid)}
\end{eqnarray}$$

The term inside the product is the ratio of the joint probability to the product of individual probability, that represents the extent to which P(aid) is enhanced by the observation of (aid_i, type_i). It is quite impossible to assume the above independence with this data, and therefore this value can exceed 1 and is no longer a probability. However, I expected it to work reasonably well in prioritizing candidates.

Each term in the above equation is obtained by counting the frequency of each aid and co-visitation of aid pairs and dividing by the total number of sessions. In calculating the co-visitation matrix, only interactions within a 24-hour period are counted, and no multiple counts are made within the same session. The period of calculation was the entire period including test data (in inference phase), and co-visitation in both directions was to be counted.

The co-visitation matrix does not hold for all pairs of aids, but only those that have many co-visitation for each aid. For aid pairs that are not in the co-visitation matrix, the ratio of the joint probability on the right side of the above formula is set to 1, so that they do not affect the score calculation.

In the actual calculation, the logarithm is taken and further weighted to the most recent action, as follows:
$$Score(aid) = \log(P(aid)) + \sum_{i=1}^{10} \frac{11-i}{10} \cdot \log \left( \frac{P((aid_i, type_i), aid)}{P(aid_i, type_i) P(aid)} \right)$$

Many other heuristics, such as adding pseudo counts and adjusting by action type, have been incorporated, but they are too complicated to mention here.

Starting with those with the highest candidate score, the top 200 were taken for training data, the top 300 for inference of test data, and then the already visited aids were added to make the final candidates.

The recall on validation of the top 200 candidates thus obtained was as follows:
- clicks: 0.697
- carts: 0.559
- orders: 0.736

### 2.2 LightGBM Rerank Model

This part is not much different from the methods already shared by others. The rerank model was trained by LightGBM (LamabdaRank), and separate models were built for each action type.

On validation, the second last week of train set (truncated) was used as training data and the last week of train set (truncated) as validation data.
When inference was made on the test data, a model trained on the last week's data (LightGBM1) and a model trained on the second last week's data (LightGBM2) were built, and their outputs were ensembled by simple average. Since the training data were completely swapped, I expected a reasonable ensemble effect, but in fact it seems that the effect was only slight.

Most of the features are based on co-visitation matrix, but each aggregation period is separate for training, validation, and test. That is, the co-visitation matrix is created for each of the three different periods, and the features are created, so they are leakage free.

The total number of features in the final model is 344, as follows:
- session features (32)
    - the number of all actions (1)
    - the number of each action (3)
    - the number of unique aids in the session (1)
    - the number of unique aids of (carts/orders) and the ratio to the above (4)
    - the last action type (1)
    - the last relative timestamp from the start of the test period (1)
    - the number of actions from the last of each action type to the last action of the session (3)
    - elapsed time from i-th last action (i=2, ..., 10) to the last action of the session (9)
    - revisit ratio of all aids by pair of action types (9)
- aid features (50)
    - count of (any/buy/click/cart/order) (5)
    - exponential decay count of (any/buy/order) (3)
    - count of (any/buy/order) in last n days (n=1~7) (21)
    - count of (any/buy/order) in last n weeks (n=1~4) (12)
    - revisit count in all sessions by pair of action types (9)
- session*aid features (12)
    - the latest action of that aid (1)
    - the number of each action of that aid (3)
    - the number of actions from last visit to that aid to the last action of the session (1)
    - elapsed time from last visit to that aid to the last action of the session (1)
    - the above two features for each action type (6)
- co-visitation features (250)
    - the number of co-visitation of aid with aid_i (i=1, ..., 10) devided by the global count of aid_i
        - any to any (both direction/oneway) (20)
        - any to buy (both direction/oneway) (20)
        - buy to any (both direction/oneway) (20)
        - buy to buy (both direction/oneway) (20)
        - type_i to any (both direction/oneway) (20)
        - type_i to buy (both direction/oneway) (20)
        - click to click (both direction) (10)
        - click to cart (both direction) (10)
        - cart to click (both direction) (10)
        - cart to cart (both direction) (10)
    - the rank of co-visitation of aid with aid_i (i=1, ..., 10)
        - any to any (both direction) (10)
        - any to buy (both direction) (10)
        - buy to any (both direction) (10)
        - buy to buy (both direction) (10)
    - global count of aid_i (any/buy/click/cart/type_i) (50)

\* "any" means the action clicks or carts or orders, and "buy" means the action carts or orders.

## 3. GNN part (by toshi_k)

### 3.1 Basic Idea

I considered using DL (Deep Learning) in this competition. Since the datasets are relatively simple, E2E approach of DL seemed like a desirable solution for me. Another advantage is that multi-dimensional interactions and outputs for clicks/carts/orders are easily designed as a DL model architecture.

DL based recommendation was initially proposed as a kind of non-linear collaborative filtering. The typical one is training AutoEncoder model and using the reconstruction methodology to evaluate missing ratings.

- Training Deep AutoEncoders for Collaborative Filtering
    - https://arxiv.org/abs/1708.01715

Although I implemented this type of method as a prototype, it didn't work well. The number of items was so large that it made input vectors ultra sparse. It also yielded the heavy requirements of GPU memory for FC (Fully Connected) layers and made hidden layers shallower and thinner.

The disadvantage of FC layers is having weights between all combinations of aids even if most of them have nothing to do with each other. After some considerations, I figured out GNN (Graph Neural Networks) can solve this issue. The graph for GNN can represents aid relations and GNN can predict attributions of aids based on the nearly connected aids.

Using GNN for session based recommendation is also reported in the below study. According to the paper, their method is developed to explore rich transitions among items and generate accurate latent vectors of items. Their experiments on two datasets including thousands of items show that their method outperforms the state-of-the-art methods.

- Session-based Recommendation with Graph Neural Networks
    - https://arxiv.org/abs/1811.00855v4

My approach is similar to the previous study. One of the biggest differences of problem setting is the number of items. In this competition, the datasets contain millions of items. To handle all items, I built a simpler workflow and installed the subgraph extraction from the global session graph.

Basically, my approach has 3 steps.
1. Construct the global graph that represents aid relations
2. Extract the subgraph from the global graph for each session
3. Use GNN to predict which aid will be taken

The conceptual diagram is as below.



In the first step, the global graph that represents aid relations is constructed. A subset of training data is used to create this graph. All transitions in this data are counted up and top P transitions (P=20, 30 or 40) from each aid are adopted as the edges of the graph. This graph is roughly corresponding to the co-visitation matrix that other participants call.

Secondly, the subgraph is extracted from the global graph for each session. The history of each session is traced and nearly connected nodes (=aids) are listed up. Closer nodes and major transitions are prioritised and hundreds aids are filtered for extraction. This process is roughly corresponding to the candidates generation other participants call.

Thirdly, subgraphs are used to train and test GNN. The last layer of GNN has three channels. They predict if clicks/carts/orders will be taken in the future of each session. The inputs of GNN are the structures of subgraphs and features of nodes and edges. More details of features and GNN model are described in the next two subsections.

### 3.2 Features

The input features for my GNN consist of "node features" and "edge features". Node features represent the characteristics of each aid. Edge features represent the relations between each pair of aids.

Basically the total number of node features is 18. Nine of them is the global characteristics of aids. These features are shared among all sessions. The other nine features represent the history of sessions. These features are calculated on the session history and different for every sessions. The list of node features is as below.

- Node features (18)
    - Global aid features (9)
        - Popularity counts (3)
        - Repeat counts (3)
        - Type transition counts (3)
    - Session history features (9)
        - Distance from the session history (2)
        - Number of counts in the session history (3)
        - Visited order features (2)
        - Visited time features (2)

Total number of edge features is 14. Twelve of them is the global characteristics of transitions. These features are shared among all sessions. The other two features represent the history of sessions. These features are calculated on the session history and different for every sessions. The list of edge features is as below.

- Edge features (14)
    - Global transition features (12)
        - Transition count not considering types (2)
        - Transition rank not considering types (2)
        - Cart-to-cart transition count (2)
        - Cart-to-cart transition rank (2)
        - Order-to-order transition count (2)
        - Order-to-order transition rank (2)
    - Session history features (2)
        - Self loop or not (1)
        - Stepped in session history or not (1)

Any combinations of multiple features and higher dimension features are not added. It was expected that such complex features were automatically captured by the representation capability of GNN.

All missing values are filled with zero and logarithmic transformation (log1p) is applied to most features for the stability of GNN.

### 3.3 Model and Loss function

My GNN has 8 GCN (Graph Convolution) layers. This implies GNN model can consider aids located within 8 steps from the history aids for prediction. In some trial experiments, 8 layers model was better than 4 layers one a little, but it was unclear more layers helped or not. 

Aside from GCN layers, my model employs non linear activation functions, skip connections, and normalization layers. These component made training faster and yielded less training loss.

As mentioned above, the last layer has three channels for clicks/carts/orders. The channels for carts and orders are connected with sigmoid functions and trained by binary cross entropy loss. The channel for clicks is connected with softmax function and trained by softmax cross entropy loss.

### 3.4 Performance boosting

I noticed that increasing the number of candidates for inference boosted the LB score. For example, my best model improved from 0.59474 to 0.59750 on public LB by increasing the candidates from 384 to 1024.

| Model | num of candidates | Public LB | Private LB |
| --- | --- | --- | --- |
| Model1 | 384 | 0.59474 | 0.59451 | 
| Model1 | 1024 | 0.59750 | 0.59719 |

Unfortunately, more candidates required more computational calculation, 1024 candidates took several days for inference. I increased candidates as much as possible and save the outputs of several models.

As the final stage of the competition, ensemble of multiple predictions is attempted. Four GNN models are used for this ensemble. Although they were trained with slightly different setting, their basic approaches were the same. Ensemble of 4 models achieved 0.59894 on public LB and 0.59874 on private LB.

| Model | num of candidates | Public LB | Private LB |
| --- | --- | --- | --- |
| Model1 | 1024 | 0.59750 | 0.59719 | 
| Model2 | 1024 | 0.59653 | 0.59613 |
| Model3 | 512 | 0.59320 | 0.59318 |
| Model4 | 768 | 0.59529 | 0.59478 |
| Ensemble of 4 models | - | 0.59894 | 0.59874 |

## 4. Team Ensemble (by toshi_k)

After we made up a team, we tried several ways to merge our predictions. We started from the simple rerank by arithmetic mean of ranks in submission files. It improved the public score 0.597→0.600 then.

The next attempt was using raw prediction to boost LB score more. We saved the raw prediction of top 50 aids of each part. The biggest issue was the outputs from LambdaRank range in any real number while the output of GNN range in (0, 1).

Since we didn't have enough time to lead the best theoretical way, we tried some ensemble method experimentally. One interesting finding was the logit transformed value of GNN seemed to have the proportional relationship with LambdaRank with constant shift.

$$ \mathrm{logit}(p^\text{Binary Prediction}) \propto v^\text{LambdaRank prediction} + C $$

The value of constant shift is different on every session_types. This may be just a brute force approximation, we estimate C for each session_type and mapped the output of LambdaRank to 0-to-1 value.

$$\begin{eqnarray}
\hat{C} &=& \arg \min_C \\{ \frac{1}{R} \sum_r^R {^Gp_r} - \frac{1}{R} \sum_r^R \sigma(^Lv_r + C) \\}^2 \\\\
^Lp_r &=& \sigma (v^L_r + \hat{C}) \\\\
\mathrm{where:} \\\\
^Gp_r &=& \text{rth output of GNN} \in (0, 1) \\\\\\
^Lv_r &=& \text{rth output of LambdaRank LGBM } \in \mathbb{R} \\\\\\
^Lp_r &=& \text{0-1 calibrated value of } ^Lv_r \in (0, 1)
\end{eqnarray}$$

After this transformation, simple weighted averaging was calculated. Since LightGBM part achieved better score on PublicLB, the weight of LightGBM is set to be larger than GNN part. We tried two patterns of weight settings and chose both of them as final submissions.

|  | weight | Public LB | Private LB |
| --- | --- | --- | --- |
| LightGBM Part | 1.000 (LightGBM) : 0.000 (GNN) | 0.60025 | 0.60008 | 
| GNN part | 0.000 (LightGBM) : 1.000 (GNN) | 0.59894 | 0.59874 | 
| Final Submission 1 | 0.600 (LightGBM) : 0.400 (GNN) | 0.60311 | 0.60307 | 
| Final Submission 2 | 0.525 (LightGBM) : 0.475 (GNN) | 0.60302 | 0.60313 |

Both of final submissions improved LB score from the best of two parts. While Final Submission 1 was the best on public LB, Final Submission 2 was the best on private LB. Even though all single models got worse on private LB, the score of Final Submission 2 on private LB is better than public LB. Out team merge and ensemble was successful in this sense.

## 5. Conclusion

Our team employed LightGBM and GNN for this competition. Each approach took different way for candidate generation, feature engineering and model design. Ensemble of two types of approaches boosted our team score a lot.

This competition gave 6th gold model to toshi_k and 7th gold to Jack. It motivate us for the further success. What we learn in this competition can be applied to not only future competitions but real world projects. It was confirmed that Kaggle is the practical platform of data science again.

Thank you for reading this to the end!
