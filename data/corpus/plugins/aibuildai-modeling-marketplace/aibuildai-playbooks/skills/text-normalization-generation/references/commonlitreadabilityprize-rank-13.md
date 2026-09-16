# [Private 13th Solution] Pseudo-labeling

Competition: commonlitreadabilityprize
Rank: #13
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257629

Thanks the community for this competition!

### Overview
- Model architectures: roberta-large, xlnet-large, electra-large, deberta-large.
- Attention head: [maunish's clrp-roberta-svm](https://www.kaggle.com/maunish/clrp-roberta-svm).
- Semi-supervised learning, Pseudo-label.
- Hyperparameters: I didn't tune much and use mostly original setup from [torch's training script] (https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-fit) since I won't have much chance competing in what I'm far behind others.

Note: I'll report my Public scores in this post since Public / Private difference is only around 0.001.

### First half
I spent the majority of first half trying different approaches (classification,  custom loss) and learning to use hunggingface with TF and Pytorch. But I didn't know how things work so all of the experiments were not reproducible and turning into classification problem basically throws all the ordinal information away so it wasn't effective.

After getting a hang of Pytorch and huggingface, I trained models TF models on train set that was shifted to the left since TF models were overpredicting on val set. Ensembling roberta-large with xlnet-large got me to LB **0.462**. Adding Pytorch models and attention head trained with [torch's amazing script](https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-fit) boosted me to **0.458**.

At this point my single best 5 fold model was only **0.468**.

### Second half
#### Pseudo-labeling
Inspired after reading [Self-training with Noisy Student improves ImageNet classification
](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf).

I used mainly [CMU's book summaries](https://www.cs.cmu.edu/~dbamman/booksummaries.html), [movie summaries] (https://www.cs.cmu.edu/~ark/personas/) and a little bit of [Mark Wijkhuizen's wikipedia abstracts](https://www.kaggle.com/markwijkhuizen/simplenormal-wikipedia-abstracts-v1). Datasets were created by cleaning and mixing different ones together. I generated pseudo-labels with my best ensemble at the time. Most datasets have **16k to 26k** excerpts.

Models were trained with 80% of train data (first fold) concatenated with the new dataset. This helped me boost my literal single model (1 file) to **0.465** on roberta-large. 

After this I spent two weeks struggling to find what makes a good dataset, I ended up with 20+ different datasets and still weren't able to figure it out due to my lack of knowledge. 

Last two weeks of the competition, I decided to use only 3 datasets that performed best and simply added xlnet-large, electra-large, deberta-large and my best score becomes **0.455** with 4 files. Adding SVR **0.454** with 4 files. Splitting the new dataset to 4 folds and ensembling deberta-large, electra-large got me to **0.452** with 8 files.

Few findings:
- Public score got worse after the first generation. I hypothesize this is due to model's learning capacity since this method leans more towards distilling and compressing model rather than the usual setup for most semi-supervised learning methods where a single model's prediction is used to generate pseudo-labels instead of an ensemble. This finding is similar to what was found in [On the Efficacy of Knowledge Distillation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf) by the authors even though it's a different technique: "Bigger models are not better teachers".
- deberta-large performed best.
- Adding noise didn't improve CV so I didn't submit.

Note: I'm not an expert in the field and there might be hindsight bias so feel free to correct me.

#### Finishing line
- After ensembling a bunch of models together, I finished with **0.450 (20th place)** on Public LB.
- Due to long training time (~2-3 hours/model), I used up to 6 Colab accounts in the last week LOL. I feel bad for this but that's the best I can do with my 8GB RAM laptop and Colab Pro isn't available where I live.

### Summary
**What worked**
- Attention head: +0.004.
- Pseudo-label: +0.015.
- SVR: +0.001.

**What didn't work**
Most of my experiments revolves around new datasets so I don't have a lot here.
- Classification: I couldn't utilize ordinal information for classification task.
- Iterative pseudo-labeling: Score gets worse from 2nd generation (my thoughts on this above).
- Post-processing: Interval mean matching, basically I try to match prediction's mean in each interval ([-0.5, 0.5], ...) with the train set. Worked okay on validation set but not LB.
