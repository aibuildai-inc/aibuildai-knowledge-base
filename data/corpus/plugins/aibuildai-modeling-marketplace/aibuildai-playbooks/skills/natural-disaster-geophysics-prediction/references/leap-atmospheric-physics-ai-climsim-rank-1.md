# 1st place solution for the LEAP competition

Competition: leap-atmospheric-physics-ai-climsim
Rank: #1
Source: https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523063

This competition was an amazing experience and a wonderful sandbox to test ideas and experiments with various techniques and architectures in-depth. Thanks to Kaggle for this fantastic platform, to Colab for providing me cheap TPU to train on, to Google as the owner of both Kaggle and Colab and for providing us with Tensorflow. When I think about it, I trained on TPU (google) on tensorflow framework (google) on Colab (google) for a competition held on Kaggle (google). It's all Google from start to end. You are amazing, and I salute you.   

I also want to express my heartfelt gratitude to the competition host @jerrylin96 . Your efforts in providing us with this amazing competition and your unwavering commitment to keeping it on track, even when facing unexpected LEAK problems, are truly commendable. Thank you for your dedication and hard work.  

Finally, thank you to the Kaggle community, especially to all the active and helpful people on the forum and in the code section. You make the experience so much better. I love you.  

This competition was a strange one. I built the best models, but there were better Kagglers than me- people who found the leak(s) and succeeded in utilizing them. I did not. I still have much to learn. I won't pretend I'm unhappy that things turned out the way they did, but the best Kagglers deserve respect. You have mine.

I know that as 1st place, and considering the significant gap in scores between my solution and the next ones, a lot of eyes would be on me to ensure I did not exploit any LEAK. Hence, I made a lot of effort to provide a complete Kaggle pipeline, from downloading the data from HF through TFRecords encoding, training, inferring, and submission, with a training example that resulted in a single model 0.79081/0.78811 public/private LB. By following my pipeline step by step, you can also construct the dataset and train your own ~0.79+ public LB model from scratch, ensuring no hiding of leaky pseudo-labels in the train set or nefarious test-set reverse engineering in inference. All of this by simply copying and running a series of Kaggle notebooks. Admittedly, there are a lot of notebooks- over 100 notebooks if you intend to download and encode to TFRecords all the data yourself- but with the massive amount of data in this competition, there is no way around it. See the complete pipeline and extra details [in my GitHub](https://github.com/shlomoron/LEAP-solution).
## Context section
[Business context](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).  
[Data context](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data).  
## 1. Overview of the Approach  
### 1.1. model

In one word, squeezeformer. This would not surprise anyone who followed some of the more similar competitions on Kaggle in the last year, in particular [ASLFR](https://www.kaggle.com/competitions/asl-fingerspelling) and [Ribonanza](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding). I used the same modified squeezeformer blocks that I saw in the [2nd solution to Ribonanza](https://github.com/hoyso48/Stanford---Ribonanza-RNA-Folding-2nd-place-solution) by @hoyso48 . The only changes I remember doing is that I deleted the first LayerNorm in the 1Dconv block and added an ECA layer. However, there may have been more minor changes that I forgot. My Tensorflow implementation was guided by hoyso48 PyTorch implementation, so once again (3rd competition in a row), I give him my thanks.  
I used 12 block models with dimensions of 256/384/512. Before the squeezeformer blocks, I have a linear dense layer followed by LayerNorm as an encoder from the input data to the model dimensions. After the squeezeformer blocks, I have a prediction head of swish dense followed by a GLUMlp block (swiGLU followed by linear dense), both with head dimensions of 1024/2048, depending on the model. For more details, check my code.  
At first, I used dropout layers, which helped greatly when the training data was in the order of 1M samples. However, after I saw comments in the forum about dropout being unnecessary, I experimented again when I scaled up to ~40M-80M samples, and it was indeed unnecessary when ensembling (although it still allows for higher scores for single models, at the cost of twice or thrice epochs). Since removing it allows for much faster training, I also chose to drop the dropout altogether. Optimizer was AdamW with half-cosine decay scheduler, max LR 1e-3, and weight decay = 4*LR.
### 1.2. Loss  
I once read that the most important part of a DL model is the loss function. I was a bit skeptical then- sure, it is important, but it's not exactly complicated to choose the appropriate loss, right? Say, if our metric is R-squared, the best loss will obviously be...MSE?
#### 1.2.1. Use MAE  
It performs better than MSE in every way- it converges faster, is better at convergent, and is more stable. If I need to choose one 'secret' of this competition for a high score, aside from the notorious leaks, it is this.
#### 1.2.2. Auxiliary loss  
We had explicit spacetime data for the train set but not for the test set. Auxiliary loss is a common practice in such cases. I used MAE on the normalized latitude/longitude and the sin/cos on the day/year cycles (if it is unclear, please look at my code).
A side note on the auxiliary loss- some people speculated, in the last week of the competition, that using explicit spacetime data, even that of the training set, is considered a forbidden leak. So, first, such use was [confirmed by the host to be legit](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/519772#2919233), as long as there is no 'hacking' of the test set (which can be done by using the auxiliary predictions to pseudo-label the test set- a thing I did not do). Second, in the last week, I trained several models without auxiliary spacetime loss, and my no-spacetime ensemble of 6 models got 0.79355/0.79092. So I win anyway; auxiliary spacetime loss is unnecessary and may not even be helpful (since my winning ensemble, while with a slightly better score, is also 13 models).  
#### 1.2.3. Confidence head  
The [3rd place solution in Ribonanza](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460403) got a suspiciously good score for a particular model. I won't explain why it was suspicious; you will need to know Ribonanza's details for the context. In any case, this suspicious model had two unique things going for it- a strange architecture and a confidence head. I still need to experiment more with said strange architecture, but for the confidense head, it was easy to try in this competition and surprisingly effective. The idea is simple- I also predict the loss of each target. I also used MAE for the confidence loss. I have one model I trained without confidence head, and it got LB 0.78945/0.78631, so I think I would also won without it. Then again, a model with the seme specifications, but with confidense head was my best model with LB 0.79159/0.78869. So, yeah. Surprisingly effective. And yes, this second model can win 1st place in this competition by itself (0.78869 private, although not 1st in public).
#### 1.2.4 Masked loss  
I masked out the targets that are zeroed in the submission or those for which we use the ptend trick (ptend_q0002_2-ptend_q0002_26).  
For those new at LEAP- please read [this post](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484) for ptend trick context.  

Final thoughts on the loss function: well, it is probably the most important part of a deep learning model. lol.  
### 1.3. Data preparation
#### 1.3.1 High-res data  
You may have noticed already that I used high-res data. Yes, it helps. Although I would have won without it- my only low-res data ensemble of five models got LB 0.79299/0.78951. But high-res definitely helped, although it sometimes requires a special soft-clipping treatment, as you will see. Also, training with high-res is less stable. Hence, I usually used larger batch sizes (1024/2048 compared to 512 for low-res only models). I blended high-res data with a ratio of 2:1 low: high. Lower than that, the performance is weaker; higher than that, the gains are small compared to the extra training time required.    
#### 1.3.2 Multiple data representation
Multiple data representations is a trick I learned from 1st solution [at ASLFR](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485). Without going into too much detail, the situation in ASLFR was that the data could be normalized in two different ways. I remember trying both ways, finding out what is better and sticking with it. Then the competition ended, and guess what? The 1st place normalized in the two possible ways, concatenated the two representations (with a few extra steps in between; read their summary for the full details) and sent it to the model.  
First, Let me separate the features that are spread over the 60-height level, which I call X_col and the features that are the same for all the levels, which I call X_col_not.  
For X_col_not, I used only one representation: the simple normalization (x-mean)/std.   
For X_col, I used three representations. The first the same as X_col_not, where I normalize each feature in each level with its own mean/std. i.e., for state_t, then we have (state_t_1-mean(state_t_1))/std(state_t_1), (state_t_2-mean(state_t_2))/std(state_t_2) etc. In my code, I call this representation x_col_not_norm (for x_col_not) and x_col_norm (for X_col).  
The second representation normalizes each feature by the total mean and std over all the levels. For state_t, we have (state_t_1-mean(state_t))/std(state_t), (state_t_2-mean(state_t))/std(state_t), etc. In my code, I call this representation x_total_norm.  
Finally, the third representation is:  

```  
x_col_norm_log = tf.where((x_col_norm-x_col_norm_min+1)>=1, tf.math.log(x_col_norm-x_col_norm_min+1),
                                    -tf.math.log(1+1-(x_col_norm-x_col_norm_min+1)))
```

This is the kind of thing that trying to explain with words would never be as clear as just looking at the code.  
After you looked at the code and understood it, you may wonder why not use:  

```  
x_col_norm_log = tf.math.log(x_col_norm-x_col_norm_min+1)
```

Why all the extra steps with the tf.cond? See, I had a problem. I calculated x_col_norm_min only with Kaggle data, and then I scaled up my code to all HF data (which have values lower than Kaggle data x_col_norm_min) but did not want to change the normalization constant because it would break the inference pipeline. Then, I would have to use different pipelines for my old and new models. Yeah, sometimes I'm a bit lazy. Proud of it. And it turned out to be an excellent choice when I included also high-res data.  
#### 1.3.3 Wind
$$wind = \sqrt{(state_u)^2+(state_v)^2}   $$
It made sense, and I wanted to include at least one 'physically justified' thing in the model (silly me, yes). It did not really help but also did not hurt the model, so it stayed. I used only the first normalization for WIND, with:  
mean(wind) = mean(mean(state_u), mean(state_v))  
And with:  
std(wind) = sum(std(state_u), std(state_v)  

All in all, the feature dimension is:  
9[col_features]\*60[levels]\*3[representations]+60[wind_levels]+16[not_col features] = 1696  
#### 1.3.4 Features soft clipping 1  
After normalization, the data has some extreme values (~±3000). This problem exists only for the first representation. The model actually handled it easily, but I preferred to play on the safe side. So, for x_col_norm and WIND, I applied the following soft clipping:  

```  
cutoff = 30
square_cutoff = cutoff**0.5
x_col_norm = tf.where(x_col_norm>cutoff, x_col_norm**0.5+cutoff-square_cutoff, x_col_norm)
x_col_norm = tf.where(x_col_norm<-cutoff, -tf.math.abs(x_col_norm)**0.5-cutoff+square_cutoff, x_col_norm)
```

#### 1.3.5 Features soft clipping 2  
In addition to the first soft clipping, I applied a second soft clipping to deal with extreme values from the high-res set. I applied the clipping on all the representations, including WIND, and after I applied the first soft clipping (1.3.3) for the relevant features:  

```
cutoff_2 = 86.0
log_cutoff = tf.math.log(cutoff_2)
x_col_norm = tf.where(x_col_norm>cutoff_2, tf.math.log(x_col_norm)+cutoff_2-log_cutoff, x_col_norm)
x_col_norm = tf.where(x_col_norm<-cutoff_2, -tf.math.log(-x_col_norm)-cutoff_2+log_cutoff, x_col_norm)
```

I chose cutoff_2 so that the soft clipping would only affect high-res data.  
#### 1.3.6 Targets soft clipping  
I did this only for the high-res targets; each target was soft-clipped if it was too extreme compared to the low-res corresponding target min/max (this differs from 1.3.4/1.3.5 in that the soft-clipping range was different for each target). In code:  

```  
rescale_factor = 1.1
if x['res'] == 0:
    y_norm = tf.where(y_norm<norm_y_min*rescale_factor, norm_y_min*rescale_factor-tf.math.log(1-y_norm+norm_y_min*rescale_factor), y_norm)
    y_norm = tf.where(y_norm>norm_y_max*rescale_factor, norm_y_max*rescale_factor+tf.math.log(1+y_norm-norm_y_max*rescale_factor), y_norm)
```
### 1.4 Post-processing  
#### 1.4.1. Downcast and Upcast  
Special care should be taken when moving between FP64 and FP32. I encoded my TFRecords with the original values in FP64. After I processed the data (see 1.3. Except for 1.3.5 which happened after the casting for no particular reason), the values were downcast to FP32 before transferring them to the model. Then I upcasted the predictions to FP64, and only then did I apply de-normalization to get the values for submission:  

```
preds = preds + mean_y.reshape(1,-1)*stds
preds[:, np.where(stds_new == 0)] = 0
preds = preds/np.where(stds>0, stds, 1)
```

#### 1.4.2. Mean for bad targets  
This is very simple:  

```
metrics = np.asarray([sklearn.metrics.r2_score(val_labels[:, i], preds[:, i]) for i in range(368)])
for i in range(len(metrics)):
    if metrics[i]<0:
        preds[:,i] = 0
```

In reality, it was eventually unnecessary because the only bad targets were those zeroed out in the submission or in the ptend trick range (see next bullet, 1.4.3).

#### 1.4.3. Ptend trick  
Obviously. If you are new at LEAP, look [here for details](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484).  

### 1.5 Validation  
My local validation included two validation sets of randomly selected samples from the low-res dataset (the same samples for all the models), each with a size of 100K samples. In addition, I randomly selected 100K samples from the high-res dataset and 100K samples from the low-res training set (i.e., the last validation set is on samples I also train on to judge overfitting better). All in all, I had four validation sets.  
When I trained on ~1M samples, the local validation was highly correlated with the public leaderboard score, but when I scaled up to all the data, it was less correlated (probably because when I used the full train set, it includes samples that are very close in time to the local validation samples and with the same latitude/longitude, so it starts to overfit even on the validation set, as opposed to the public test set that includes samples from entirely different year than those that exist in the train set). For example, I could push my local validation score to ~0.8, but on the public LB, it will get ~0.785 and be lower in score than a model that achieved, say, 0.795 but trained for fewer epochs. So, for the final models trained on all the low-res data from HF (and those I trained also on the high-res data), I validated against the public LB to get the hang of good overfitting range, and ensembling validation was done directly against the public LB. Thus, my ensemble is slightly overfitted to the public LB, although I acted in ways that reduce said overfitting (equal blending weights and including 'less successful' models if I judged them to be successful enough and anticipated them to be good based on the parameters I used). If it sounds a bit like black magic- yes, it is! Deep learning is sometimes a science, sometimes an art.  

## 2. Details of the submission  
### 2.1 Ensembling  
My winning ensemble included 13 models, each a bit different (see full details in my GitHub, 'The steps to reproduce my solution' bullet 5). The best model (11) was LB 0.79159/0.78869, and the worst (2) was LB 0.78795/0.78388. Both were best/worst both in public and private LB. The full ensemble was LB 0.79410/0.79123. In addition, my low-res-data-only ensemble of 5 models has LB 0.79299/0.78951, and my no-spacetime-auxilliary-loss ensemble of 6 models has 0.79355/0.79092. When you read my solution, you may have tried to find out the 'secret sauce' that got me the 1st place, but it really was the combination that made the difference. Every single technique that I used, I think I could still get to 1st place without it.  
### 2.2 The helpful techniques
This is a short summary of the methods I wrote about in-depth above: Squeeseformer, wide GLUMlp prediction head, no dropout, MAE, auxiliary spacetime loss, confidence head, masked loss, multiple data representation, high-res data, features and targets soft-clipping and careful downcast/upcast.
### 2.3 What didn't work  
Various model architectures (pure transformer, other 1Dconv/transformer combinations, Unet, dropout, smaller models, larger models, other optimizers), in short, many less optimal hyper-parameters. Log-normalization (i.e., log(x), not my log(1+x) representation which deals with different issues). MSE, MSE/MAE various combinations, weighted loss function (check my Ribonanza solution for details), levels/features masking, predict the year as additional auxiliary loss, and probably other not-very-important things that I don't remember already.  
### 2.4 Hardware
I trained on Kaggle and Colab TPU and inferred on Kaggle P100 GPU. Compute-wise, my experiments and training cost at least 200 bucks in Colab compute units and probably less than 300 bucks in total. If it sounds like a lot compared to your 'free' personal machine, please consider the electricity cost of using an RTX4090...
### 2.5 Bonus: confidence is key
Since I already had confidence predictions simply because the confidence head improved the model, I took a step further out of curiosity and checked if I could get a higher score by discarding 'low-confidence' samples and by how much I could improve. Here is a graph (only for one model, not the full ensemble) that shows the R-squared as a function of the fraction of remaining samples after I discarded the most 'low-confidence' ones.  

As you can see, by discarding ~0.1 of the samples, I can already get to a ~0.83+ score. This is not very surprising: if you check the data, you will see that certain months and locations have much lower scores, so a higher score can simply be achieved by discarding said months/location. However, discarding by confidence may be more precise and efficient. It can be interesting to research further, especially given this specific problem, since we can replace the simulator calculation with model predictions and then send back the low-confidence samples to the simulator.  
As a side note, I also tried ensembling by confidence (i.e., giving lower weights to low-confidence predictions), but my preliminary research did not show promising results. Although it WAS preliminary- I got to it only on sunday of the last week of the competition, and it was either researching it more in depth or watching the Euro 2024 finals...(congrats Spain!)  

## 3. Sources
This is my favorite part. Thank you all the resources that have helped me!  
[Ptend trick](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484) and also [here originally](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/499896#2791290) by @sakvaua and @jano123 .
[Ribonanza 2nd solution](https://github.com/hoyso48/Stanford---Ribonanza-RNA-Folding-2nd-place-solution) for Squeezeformer architecture guidance and insights, by @hoyso48 .  
[Ribonanza 3rd place solution](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460403) for confidence head method, by @dankrstev .  
[ASLFR 1st solution](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485) for the multiple data representation method, by @darraghdog and @christofhenkel .  
[Dropout is unnecessary](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/514020#2884414) and [also here](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/514020#2885043) by @sakvaua and @ymatioun .   
In addition, I used the low-res and high-res data [from HF](https://huggingface.co/LEAP).  
And, of course, [my GitHub](https://github.com/shlomoron/LEAP-solution) with links and instructions for a fully reproducible solution on a Kaggle pipeline.
