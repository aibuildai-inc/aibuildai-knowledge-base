# 23rd Place Solution: You don't need heuristics or expensive GPU

Competition: hpa-single-cell-image-classification
Rank: #23
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238512

[**UPDATE**]:
For the latest version of the solution write-up, please check [**the GitHub repo README**](https://github.com/SamusRam/hpa-single-cell/blob/main/README.md). There, all the graphics should be present, but I am not maintaining external hosting links for graphics in the current post. Please, feel free to drop me a personal email via Kaggle or raise a GitHub issue in case of any questions. Thanks!

---------------------------------------------------------------------
Hello, everyone!

First of all, I'd like to congratulate the winners! It's a privilege to learn from your solutions! 

Secondly, I'd like to thank the organizers of the competition! Thank you for your kind, attentive and responsive approach on forums!
And thank you, all my fellow HPA competitors! 😊Thanks to all of you it felt like an awesome Team of amazing like-minded enthusiastic colleagues. As Darek @thedrcat has put it nicely, I wish we would meet and celebrate. Hopefully, it'd happen during some offline KaggleDays hackathon in the future 😉

# Solution
## Intro
Analogously to [the post](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/232396) by Paweł @narsil , my approach was **very much data-centric** and **theoretically rigorous**.
 
The core of my solution was **de-noising**, including de-noising based on graph Laplacian regularization theory. De-noising itself allowed me to achieve a competitive position **without compute-intensive large models**, **without heuristics** like ranking, multiplication of predictions, etc.

## De-noising
[IMAGE](https://drive.google.com/file/d/1RDrcZ9_boh4u6gTX6O5ObETK7TpSl3ns/view?usp=sharing):


I managed to implement the computationally efficient graph signal denoising from [this paper](https://ieeexplore.ieee.org/document/7450177). My Python implementation can be found [here](https://github.com/SamusRam/hpa-single-cell/blob/main/src/denoising/graph_denoising.py). The algorithm from the paper enforces label sparsity so that each object would have a single confident label after the de-noising. Therefore, outputs of the de-noising were not used directly. I focused on the group of cells having the highest mode in the de-noised soft labels. E.g., I've added around 500 cells from the highest de-noised labels:
IMAGE:


## Negative label
~~I estimated negative label probability under the assumption of labels independence:~~
$$ P_{neg} = \prod_{c\ in\ patterns} (1 - P_{c}) $$.

~~Estimating P_{neg} this way instead of trying to predict the negative label boosted my public LB score from 0.517 to 0.523.~~

*Update*: I noticed that by mistake I did not include the independence-based estimation into my final submission. In the submission corresponding to my private LB position I estimated negative label probability as 
$$ P_{neg} = \min_{c\ in\ patterns} (1 - P_{c}) $$. 

Such a way to estimate P_{neg} was worse compared to the independence-based estimation by 0.0007 on public LB. But it turned out to give a result by 0.0003 better on the private LB, luckily.

## Segmentation
I implemented the removal of border cells. I also modified the removal of small nuclei, doing it based on medium nucleus size instead of the hardcoded threshold. These modifications boosted my public LB score from 0.523 to 0.527.

## Simple model
 
 5 folds, DenseNet121 by Shubin @bestfitting, referenced in [the Nature paper](https://www.nature.com/articles/s41592-019-0658-6) and [on forums](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/214925). Thank you for your awesome work, Shubin @bestfitting !

Data-related challenged seemed to be of higher priority, I focused on those and had no time/hardware to switch from DenseNet121. 

*For reference I checked that on ImageNet DenseNet121 is currently ranked 325th with top-1 accuracy of 75%, while state-of-the-art is 90% ([source](https://paperswithcode.com/sota/image-classification-on-imagenet)). Other successful teams used EfficientNets, Swin transformers which achieve up to 87% on ImageNet.*

## Strict avoidance of heuristics
The hosts mentioned that the hidden test set was designed to have high single-cell variation. For this reason, I fought the temptation of heuristical re-ranking based on image-level labels, etc., even though the community reported that it helped on the public test set. Regarding the private LB placement, it seems that I've been unnecessarily cautious.

## Splitting cells
I've invested quite some time into modifications of the HPA-Cell-Segmentation to split crowded cells, e.g. in tissues, but it didn't provide a public LB boost. The work-in-progress splitting algorithm can be found [in this branch on GitHub](https://github.com/SamusRam/HPA-Cell-Segmentation/tree/closer_to_orig)

## Masking cells
When predicting a single cell, I masked all surrounding cells out, so that in the high SCV scenario predictions would not be influenced by neighboring cells in the bounding box. I later realized that it might make it harder for the model to spot later phases of mitosis. Yesterday I quickly tried to create a separate binary classifier for the mitotic spindle, where I included the surrounding cells in the extended bounding box. But the quick experiment with including the mitotic spindle classifier didn't provide a score improvement.

## Source code
.. is in [this GitHub repo](https://github.com/SamusRam/hpa-single-cell). The high-level entry point would be `orchestration scripts`.
