# 26th Place solution - Record the first kaggle competition ,failed to gold (unlucky)

Competition: feedback-prize-english-language-learning
Rank: #26
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369691

First of all, thanks to the host for an interesting competition and congratulations to all the winners!
Since I am Chinese and not very good at English, there may be some inappropriate words, please bear with me
# **Overview**
The experiment is divided into two parts, one is the debugging of the basic model (adding various tricks), and the other is to distill the knowledge of the better ensmble model and then go to the original data to continue training and fine-tuning.
# **Base model**
|Experiment Number| model type|  max_length    |  fold  |  cv score  |
| --- | ---|
1 |    deberta-v3-base	|768	      |10	    |0.4521
2|	deberta-v3-base	|768	      |5	|0.4519
5|	deberta-large	        |512	      |5	|0.4516
6|	deberta-xlarge	|512	      |5	|0.4536
10|	deberta-v3-base	|512	      |5	|0.4502
13|	deberta-v3-large	|512	      |5	|0.4512
15|	deberta-v3-large	|1024     |5	|0.4506
17|	deberta-v2-xlarge	|1024     |5	|0.4546
## **what worked**
**● [Worked] FGM**

○ Result: cv +0.001-0.002

○ Reasoning and context: Starting from epoch 0 with epsilon=0.25 for confrontation attack, too low or too high will not work well (joining fgm confrontation learning means that your training time will basically double)

**● [Worked] layernorm**

○ Result: cv +0.0005-0.001

○ Reasoning and context: Adding a layernorm layer in front of fc can also make the effect better.

**● [Worked] Attention Head**

○ Result: cv +0.0005

○ Reasoning and context: Comparing [CLS], mean pooling and attention head, there is basically no difference in the effect of the former two, mean pooling may be better, and the best effect is the attention head.

**● [Worked] Multi-sample Dropout**

○ Result: cv+0.0001

○ Reasoning and context: The effect of Multi-sample Dropout on the base is not very good, causing the training loss to oscillate, but the effect on the v3-large model is good, and it can prevent the large model from overfitting.

**● [Worked] LLRD**

○ Result: cv+0.001

○ Reasoning and context: After comparing the learning rate of 0.8-0.9 for each layer, and setting the learning rate by dividing the number of layers into blocks (it is better to divide 3 or 4 layers into a group), the latter brings me a greater improvement in cv .

## **What Didn't Worked**

**● [Didn't Work] AWP**

**● [Didn't Work] other heads**(maxmeanpool、weight pool、contact different layers, etc.)

**● [Didn't Work] MLM** (pre-training does not work well on fb3 and fb2)

**The ensmble of the basic model uses the simplest optuna method to adjust the weights of different models (at that time, I thought about using the second-stage ensmble model after knowledge distillation, such as: svr, lasso, linear, gbdt, etc.)**
# **Knowledge distillation and continued fine-tuning:**
● dataset:Data after FB1 removes FB3

**● stage 1**

○ Label the FB1 data predicted by the previous ensmble model.

○ Use the pipeline training of the previous better model

○ Increase epoch (10 I set)

○ Use fb3 as the verification set to save the best model

**● stage 2**

○ Continue to fine-tune the trained single model on the data on fb3 (5fold)

○ Low learning rate (5e-6)
|Experiment Number	|model type	|max_length	|cv score	|pr score	|lb score	|head
| --- | --- |
|exp0050	|deberta-large	|512	|0.4396	|0.436596	|0.440218	|mean
|exp0051	|deberta-xlarge	|512	|0.439	|0.436895	|0.437808	|mean
|exp0058	|deberta-xlarge	|512	|0.4384	|0.436683	|0.436752	|attention
|exp0065	|deberta-v3-large	|1024	|0.4387	|0.436483	|0.43784	|mean
|exp0068	|deberta-v3-large	|1024	|0.4385	|0.436954	|0.437924	|attention
|exp0071	|deberta-v3-large-squad2	|1024	|0.4395	|0.436845	|0.439193	|mean
|exp0074	|deberta-xlarge-mnli	|512	|0.4394	|0.436168	|0.438751	|mean
|exp0072	|deberta-xlarge-mnli	|1024	|0.4398	|0.435758	|0.438707|attention

It's a pity that I didn't choose the model with the best pr performance in the end. I only chose (58, 51, 65, 68), and was deceived by LB, because the method itself has leaks, so I didn't believe in cv very much, but Choose the case where the cv is as low as possible, and the lb performance is also good.
The final model ensmble tried different stack technologies (svr, lasso, linear, gbdt), and finally chose optuna with the best effect, only adjusting the weights of different models (I also feel very strange at this point, the conventional ensmble technology doesn't work)

# **Important Citations:**
[CommonLiit Readability Prize 1st place solution](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/257844)
[FB3 / Deberta-v3-base baseline [train]](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
# **Team Members:**
[@DJcheng_gosolo](https://www.kaggle.com/djchenggosolo)
[@paradoxg](https://www.kaggle.com/koukinn)
[@Tian](https://www.kaggle.com/tiandaye)
