# 33rd Solution Writeup and Discussion on GST

Competition: predict-ai-model-runtime
Rank: #33
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456579

First of all, thanks Kaggle and the competition host for hosting this exiting competition and congrats to all the winners. I would like to share my solution (though not that good) mainly from the perspective of [**G**raph **S**egment **T**raining (**GST**)](https://github.com/kaidic/GST). The code is released [here](https://github.com/JiangJiaWei1103/Google-Fast-or-Slow).

## Overview
* Data cleaning and preprocessing
    *  Add new features following instructions [here](https://github.com/google-research-datasets/tpu_graphs/tree/main#graph-feature-extraction).
    * Drop constant and quasi-constant features.
    * Label encode features related to `shape_element_type_is_X`.
    * Log transform features with max value greater than 20.
* Model architecture
    * Train all models in the **early-join** manner (*i.e.,* fuse node and config features before getting the whole graph context).
    * Use `SAGEConv` as the GNN block.
* Training strategy
    * Use **GST** (with history embedding table and stale embedding dropout) to train *layout* models in a collection-specific manner (*i.e.,* one model per collection).
    * Sample a subset of configurations to train models per iteration.
* Experimental Setup
    * Loss criterion: `PairWiseHingeLoss` for *layout* and `ListMLE` for *tile*
    * Optimizer: `AdamW` with base learning rate `1e-3` (I decrease lr when increasing #epochs)
    * Learning rate scheduler: Cosine schedule without warmup
    * Checkpoint: Always pick the model at the last epoch

## Data Cleaning and Preprocessing
I process the data by a simple four-stage workflow. Firstly, I find out that some of the features are constant among all the datasets, which can be viewed as the redundant dimensions and dropped directly. Also, those with constant ratio like above 0.999 (*i.e.,* quasi-constant) are thrown away. Then, I label encode the remaining dimensions related to `shape_element_type_is_X`, which can be represented with a dense embedding. Finally, considering features can span a wide value range (also, some outliers exist), I simply use `np.log1p` to log transform features with max value greater than 20.
After processing, the node feature dimension drops to 116 and 50 (89 and 33 without new features added) for *xla* and *nlp*, respectively.

## CV Scheme
Considering there are only ~4 graphs and 8 graphs for *xla* and *nlp* evaluated on public LB, I try to enlarge the validation set by splitting train+val stratified on runtime, which can somewhat balance the intrinsic graph properties (I explore relationship between graph stats and runtime in [this notebook](https://www.kaggle.com/code/abaojiang/google-fast-or-slow-detailed-eda)). However, I don't think it's much different from just using the official train-val splitting.

## Model Architecture
[[Screenshot-2023-11-20-at-15-10-31.png]](https://postimg.cc/mtC0KZzX)
The figure above illustrates the overview of model architecture, where \\(d_n \\) and \\(d_c \\) denote the node and config feature dimensions. And, \\(L \\) is the number of graph convolution layers.
Since my first submission on 22nd, Oct, I use early-join to fuse the node and config features. After experimenting with different GNN blocks (*e.g.,* `GATConv`, `GATv2Conv`, `GINConv`), `SAGEConv` always outperforms, so I stick to it till the end. Also \\(L \\) is always set to 3. To be honest, there's no fancy design in my model architecture. Hence, I want to talk more about the training strategy.

## Training Strategy - **G**raph **S**egment **T**raining (GST)
Considering the memory limitation, I quickly decide to choose off-the-shelf **GST** as my training framework. As there exists some unsolved issues in the official implementation of **GST**, I rewrite the pipeline without [GraphGPS](https://github.com/rampasek/GraphGPS).
The main concern with **GST** is that the training loss increases as the training process progresses, but validation performance still improves over time. After fixing the \\(\eta \\), the weight for each graph segment, for final sum pooling, the training loss decreases normally as shown below (special thanks to @dsfhe49854 's  analysis [here](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/448367#2497447)),
[[Screenshot-2023-11-20-at-17-50-00-Weights-Biases.png]](https://postimg.cc/jCyB81QT)
Let's see how \\(\eta \\) is derived in the original paper. Let \\(n \\) be the number of segments for one graph and \\(k \\) be the number of segments to be trained per iteration. Also, select \\(p \\) as the dropout ratio of **stale embedding dropout**. Assume we sample only one segment for training per iteration(*i.e.,* \\(k = 1\\)), as described in the paper. The weight \\(\alpha \\) of the trained segment can be derived as follows, 
$$
(n-k)p + k\alpha = n
$$
$$
\alpha = (1-p)\frac{n}{k} + p
$$

The logic behind the scene is that the final runtime estimation is the **sum pooling** of runtimes of all segments. Considering some segments are dropped with probability \\(p \\), we need to increase the weight of the trained segment for compensation. However, the problem is that most of the entries in historical embedding table are zeros. Therefore, in early epochs, the objective can be approximated as,
$$
\hat{y} = \alpha \hat{y}_{i} ,
$$

where \\(\hat{y} \\) is the predicting runtime of the current graph and \\(\hat{y}_{i} \\) is the predicting runtime of the segment \\(i \\) of the current graph. What's interesting is that I observe the **unfixed \\(\eta \\)** always leads to better generalizability compared with the fixed one. Also, if the model is trained with sufficient number of iterations, the training loss actually goes downward (the red line turns the direction at ~100 epochs).

## Experimental Results
Following table shows the local CV scores of my final submission.
| Collection | CV | 
| --- | --- | 
| *tile* | 0.9551 | 
| *xla-default* | 0.3188 |
| *xla-random* | 0.5569 |
| *nlp-default* | 0.5053 |
| *nlp-random* | 0.8845 |

## What Didn't Work for Me
* Use other GNN blocks (*e.g.,* `GATConv`, `GATv2Conv`, `GINConv`)
* Retrain models on the whole dataset
* Finetune *default* using the pretrained weights from *random*
    * Freezing different parts of network makes no difference.
* Segment graphs with other strategies (*e.g.,* Metis)

## Conclusion
It's not good to stick to only one method (**GST**) for all implementation, I should have explored other potential solutions like all amazing writeups I've digested so far. Though the result isn't that promising this time, I will keep progressing and learning from the top-tiers. This journey never stops! Thanks for your patience!
