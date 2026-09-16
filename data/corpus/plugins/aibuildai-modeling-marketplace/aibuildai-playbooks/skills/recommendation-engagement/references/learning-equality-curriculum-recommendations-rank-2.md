# 2nd Place Solution

Competition: learning-equality-curriculum-recommendations
Rank: #2
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/395110

Thanks to Kaggle and the hosts of this competition, in particular to Jamie Alexandre for all of his helpful comments to all asked questions in the discussions and his great Data Exploration Notebook. Especially the code for traversing the topic tree was very helpful in the beginning.

I spend quite some time during Christmas vacations on this competition and then did a longer break of over one month. So, in the end the finish was quite intense and submissions for Leaderboard and Efficiency Prize at the same time felled like participating in two different Competitions and quite to less Submissions for only one week left. 

#Solution: Single Stage Retrieval based on Cosine Similarity
My solution is based on a simple one stage approach to create **embeddings** for **Topic** and **Content** and calculation the cosine similarity for the retrieval task, without any further post-processing or second stage reranking. I will describe all the pieces of the puzzle to achieve high scores with such a relatively simple model.

## Validation Split:
**10 Fold CV** split based on minimizing the overlap of content relations between different folds. In this competition we have to split the topics into folds and all topics can have multiple attached content to it. Throughout this is a **n x m** relation of **Topic x Content**, it is difficult to create perfect splits that are aligned with the leaderboard.

During CV-Split creation I tried to minimize the overlap of having the same content in different Folds, by creating 10 buckets and add topics to the bucket were all attached content of that topic creates the least overlap having the same attached content in different buckets. At least for fold 0 of my 10 Folds the alignment to the public leaderboard was quiet well. Whereas my Fold 2 is closer to the private leaderboard as it seems. I only used Fold 0, 1, and 2 for offline validation and results were always correlated to the leaderboard so I don’t care if they are perfectly aligned towards the score.

Here is one example trained for only 32 Epochs with nearly perfect alignment of fold 0 towards public lb. `Selected` means here the average count of selected content per language over all topics in that fold and the `F2`, `Precision` and `Recall` of that single model are displayed for all languages. 

```
-------------------------[Model: sentence-transformers/LaBSE]--------------------------

---------------------------------------[Epoch: 32]-------------------------------------
Epoch: 32, Train Loss = 1.064, Lr = 0.000050

----------------------------------[margin: th 0.160]-----------------------------------
Calculate Scores
en  Score: 0.65911 - Precision: 0.59954 - Recall: 0.739 (2806x65939) - selected: 7
es  Score: 0.71289 - Precision: 0.61102 - Recall: 0.838 (1177x30844) - selected: 6
pt  Score: 0.78237 - Precision: 0.69215 - Recall: 0.862 (343x10435) - selected: 7
ar  Score: 0.51809 - Precision: 0.46233 - Recall: 0.663 (318x7418) - selected: 7
fr  Score: 0.59613 - Precision: 0.59448 - Recall: 0.650 (304x10682) - selected: 9
bg  Score: 0.68063 - Precision: 0.60765 - Recall: 0.751 (242x6050) - selected: 8
bn  Score: 0.15228 - Precision: 0.09667 - Recall: 0.211 (237x2513) - selected: 9
sw  Score: 0.69321 - Precision: 0.64327 - Recall: 0.763 (209x1447) - selected: 6
gu  Score: 0.76149 - Precision: 0.66632 - Recall: 0.834 (181x3677) - selected: 6
hi  Score: 0.63803 - Precision: 0.58325 - Recall: 0.744 (138x4042) - selected: 9
it  Score: 0.87791 - Precision: 0.85495 - Recall: 0.906 (73x1300) - selected: 4
zh  Score: 0.63350 - Precision: 0.54224 - Recall: 0.740 (68x3849) - selected: 10
mr  Score: 0.69542 - Precision: 0.57128 - Recall: 0.898 (24x999) - selected: 12
fil Score: 0.72123 - Precision: 0.68860 - Recall: 0.778 (23x516) - selected: 7
as  Score: 0.58904 - Precision: 0.53932 - Recall: 0.644 (13x641) - selected: 5
my  Score: 0.71483 - Precision: 0.71825 - Recall: 0.842 (12x206) - selected: 4
km  Score: 0.91160 - Precision: 0.88671 - Recall: 0.942 (11x505) - selected: 5
kn  Score: 0.63651 - Precision: 0.55926 - Recall: 0.722 (9x501) - selected: 9
te  Score: 0.86664 - Precision: 0.73492 - Recall: 0.968 (7x285) - selected: 13
or  Score: 0.81583 - Precision: 0.69889 - Recall: 0.900 (5x326) - selected: 11
ta  Score: 0.76419 - Precision: 0.52095 - Recall: 0.967 (5x216) - selected: 6
ur  Score: 0.40010 - Precision: 0.31326 - Recall: 0.586 (5x245) - selected: 17
pnb Score: 0.87594 - Precision: 0.83333 - Recall: 0.938 (4x184) - selected: 8
ru  Score: 0.66330 - Precision: 0.63704 - Recall: 0.725 (3x188) - selected: 12
pl  Score: 0.82159 - Precision: 0.99061 - Recall: 0.792 (3x319) - selected: 30
swa Score: 0.08696 - Precision: 0.08696 - Recall: 0.087 (3x495) - selected: 23
tr  Score: 0.54843 - Precision: 0.87778 - Recall: 0.518 (3x225) - selected: 9
---------------------------------------------------------------------------------------
CV Score:  0.65414 - Precision: 0.58710 - Recall: 0.743
---------------------------------------------------------------------------------------
Public Score: 0.65069
Privat Score: 0.69607
---------------------------------------------------------------------------------------
```
## Input Data:
No use of special Token for separation instead, I use the `#` as seperator.

### `Topic:`
**Title `#`  Topic-Tree `#`  Description**
The Topic Tree is reverse ordered and the same separator `#`  is used so we end up with:
**Title `#`  Parent `#`  Grandparent `#` … `#`  Description**

### `Content:`
**Title `#` Description `#` Text** (cut to 32 based on white space splitting) 
If for example **Description** is empty the model will see as input: **Title `#`   `#` Text**

## Training:
I used only **Transformer Base Models** with a **max. sequence length** of **96** tokens for both topic and content. 

The first and maybe also the most important part of the puzzle is the use of the **InfoNCE Loss** as symmetric contrastive loss-function. Cause we have here a **n x m** matching problem it is tricky when using this loss function due to high intersection of topics and content. 



As can be seen in the above visualization we want to have matches only on the diagonal of the similarity  matrix during loss calculation, cause when using Cross-Entropy in both directions **Topic->Content** and **Content->Topic** the labels are just:

**labels = torch.arange(len(logits))**

But if we have two topics in the same batch that might share the same content we end up with high similarities besides the diagonal of the matrix whereas the label only says the high similarity on the diagonal are a correct match. On possibility to circumvent this problem is using label smoothing for the cross-entropy loss. But, having topics with related content in the same batch is simply noise for the model and should be avoided. 

The simplest way in Pytorch when using DP and not DDP is to write your own custom shuffle function and set shuffling in the data-loader to false, what means noting else than using a sequential sampler. 

My own custom shuffle function simply calculates before each epoch the composition of the batches and avoids sampling topics with related content in the same batch. Additionally after each epoch predictions for the whole training data are calculated to detect for each topic content that we miss and content that would be incorrectly assigned to that topic. 

- **Missing Content** is stored in a list and the specific pair (topic, content) gets oversampled during shuffling. 

- **Wrong Content** retrieved for a topic is more difficult to solve. Let’s say we have topic `t1` with the wrong content `c1` for that topic with a currently high similarity to that content. Now what we need is another pair `(t2, c1)` of our ground truth in the same batch, to push away `c1` from `t1` cause when using the InfoNCE loss all other **N-1** contents in the batch are negatives for that sample. This sampling strategy is highly effective cause we increase the margin to all negative picks during the next epoch. Or at least we try, because if adding `t2` would lead to a conflict based on related content (noise) it is rejected for adding in the same batch. I do this for a specific topic up to a max. number of 128 hard negatives for that sample, by a training batchsize > 768.

With that shuffle/sampling strategy we end up with batches without conflicts in related content or ambiguities, so we have only a correct match on the diagonal line and lots of near but incorrect content (hard samples) in the batch. 

## Language Switching:
I translated the most common languages into each other for additional training data. But just adding this data to our training would be a disaster when using InfoNCE, cause if we translate for example an topic and content item from English into French and a quite similar topic and content already exists in the original French content we will end up again with noise during our loss calculation.  So what I did is a simple switching strategy after each epoch, shown in the next picture.



So we can use the `correlation.csv` without having any trouble of ambiguities of translated to original content during loss calculation. Of course translation is never perfect and this alone creates some noise and further more we change the distribution of the training data. That’s the reason a switch is only used every second epoch and only between the languages en, es, pt and fr.

Using language switching seems to bring a score boost of up to **0.01 – 0.02** what is not that much as expected maybe due to the noise this introduces during training. 

## Knowledge Distillation (Efficiency Prize):
Nothing special just used my pre-trained models as teacher, drop half of the transformer layers and train a second time using just the MSE-Loss. Weights of the student are initialized with the weights of the pre-trained teacher model. Distillation leads only to a slightly drop in performance when using 6 Layers, so the sweet spot for me was not dropping more layers, but of course this is always a trade of speed vs. accuracy.



## Quantization + JIT-Trace (Efficiency Prize):
I used Torch Post Training Dynamic Quantization what is far from optimal. Unfortunately, there is no pre-installed huggingface optimum in the CPU kernel so using Intel Neural Compressor or OpenVINO needs offline installation, and if every second counts it makes not much sense to waste time with installation of additional packages. Maybe should have tested the ONNX runtime but run out of time. 

Unfortunately FX Graph Mode Quantization of Pytorch did not worked out for me with the huggingface models, otherwise I would have tested quantization aware training for better results. 

I ended up with just using `Eager Mode Quantization -> Post Training Dynamic Quantization` and compiling into a jit-traced model. This leads to a performance drop of around **0.01 – 0.02** on my cv-scores but increases the throughput and lowers the execution time. 

If using `qint8` also on the Feed Forward part of the transformer on the intermediate up sample and output layer, the score drop is even higher so I ended up in only using `qint8` on the attention layer.



## Optimal Threshold:
Last but not least but the last part of the puzzle is using an dynamic threshold instead of a static one. I can not say if this would work also for other models but I use this calculation after each epoch to find the hard negatives for the next epoch and also during inference and it gives me a boost of **0.02** in Score or even higher. The basic idea is described in the following picture. 



The biggest advantage was, that this is much more stable than finding an optimal static threshold. Every model I trained before leads to a different optimal static threshold. With using this dynamic calculation the results was always the same and values for margin were always between `[0.14 : 0.18]` on different folds. Using a higher margin leads to better recall by losing precision and because of using the F2 Score, perhaps I should have tried to submit higher values as well. During training I use **0.16** as margin, for my best submission **0.18** is slightly better, but the difference is not that high. 

## Results:
Cause the contrastive training seems to be rock stable and showing absolutely no signs toward overfitting I trained all those models for **40 epochs** and just used the final checkpoint. My experiments on fold 0-2 shows always, that no matter what checkpoint the difference in the last epochs to the best epoch is on the third digit, so it is save to train on the whole training data without any validation holdout. 

- For Efficiency Prize I used an ensemble of only two models, cause a second model leads to the biggest jump in scores by only increasing the runtime to **23 minutes**.
- For the Leaderboard I am using an **ensemble of 5 models** with a runtime of only **9 minutes** on the **P100 Instance**, what is also quite fast. 




## What did not work out:
- Using different margins for each language seems to lead to overfitting with worse results on the leaderboard.
- Cross-Encoder as second stage. As others also discovered, if the retrieval model becomes better, it is more likely that the second stage becomes useless or even leads to errors and overfitting. 
- Using T4 instance. Don’t get me wrong, it is great that Kaggle increased for CPU Instance the RAM limit and introduced the 2xT4 option but I struggled with the RAM limitations on that Instance if using Pytorch DP so I switch back to P100 were I had not such problems. 
- Training monolingual only on English. Even with my increased training samples due to translation of all Spanish, Portuguese and French Topics and Content into English, it does not show better results when using for example an `all-mpnet-base-v2`, what is monolingual,  instead of using all data and train a multilingual model.   

## Code:
* [Training (GitHub)](https://github.com/KonradHabel/learning_equality)
* [Inference: Leaderboard Prize (Kaggle)](https://www.kaggle.com/code/khabel/2nd-place-learning-equality-leaderboard-prize)
* [Inference: Efficiency Prize (Kaggle)](https://www.kaggle.com/code/khabel/2nd-place-learning-equality-efficiency-prize)

## Data:
* [Translated Data](https://www.kaggle.com/datasets/khabel/learning-equality-language-switch)

## Checkpoints:
* Checkpoints trained on all training data are available in the inference notebooks [[Link]](https://www.kaggle.com/code/khabel/2nd-place-learning-equality-leaderboard-prize/input), [[Link]](https://www.kaggle.com/code/khabel/2nd-place-learning-equality-efficiency-prize/input)
* Checkpoints trained on fold 0 for offline eval [[Link]](https://drive.google.com/drive/folders/102N-wVRLhzf9d5IL3r1HFdYCsCZqcNQF?usp=share_link)
