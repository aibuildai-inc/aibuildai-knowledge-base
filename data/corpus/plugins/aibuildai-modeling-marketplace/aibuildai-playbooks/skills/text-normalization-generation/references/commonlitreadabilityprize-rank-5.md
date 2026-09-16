# 5th place solution

Competition: commonlitreadabilityprize
Rank: #5
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/259729

Thanks Kaggle and CommonLit for hosting this competition I learnt a lot in my first NLP competition.
And also great thanks to @rhtsingh (I learnt a lot about NLP from your notebooks.), @leighplt (I use your notebook as initial baseline.) and @konradb (I use features from your notebook).

**CV**
- I use k fold without shuffle since I note that the training set is not shuffled and the excerpt seems comes in batchs of different sources. I think the test set might also composed of excerpts from other sources and it is well known that test set has more modern texts. So I think use k fold without shuffle might better measure the 'out of distribution' generaliztion ability.
- According to my early stage experiments CV without shuffle usally get cv score around 0.01 higher than shuffled ones. As LB scores are average of 5 models (I use 5 fold) and CV is given by oof prediction of 5 models but one model each time so I expect CV score to be higher than LB scores.
- Although I think k fold without shuffle seems to be better but I didn't systematically check whether it is superior than shuffled ones or even further group by bins.

**Dropout**
- Model performance seems to have a gap between .train() and .eval() mode when apply dropout in this task.I think this might comes from potential bugs that I can't identify or the variance difference  between .train() and .eval() of some layers. The scaling factor in dropout in general don't guranteen the layer to have same variance and 'Readbility' sounds like something that will related to 'variance'/'perplexity'/... Of course this might not be the real reason.
- I was not able to solve this problem thoroughly. I use dropout in my models but use .train() to inference, this give me reasonable performace and superior to the zero dropout setting in my case. Due to randomness from dropout I average the predictions from same model but with different seed while inference, this gives me further boost.

**Other hyper-parameters that worked for me**
- Adamw 5 epochs
- cosine learning rate scheduler
- change learning rate exponentially across layers
- re-initialize some layers
- gradient accumulation for larger batch size (I only tried batch size 32 with base models and 16 with large models due to notebook time limit)
- linear head followed by mean pooling on last hidden state (some times I use two linear layers for output but mainly due to I want to keep in track with one of my ealier experiemnt, perfromance of two or one linear layer is not much different.)

I included one model with ITPT in final solution but just for diversity it didn't work for me in terms of both CV and LB.

**Single models for final inference**
(score without dropout average)
- F2    roberta large         CV 0.4839   public LB 0.470  
- F17   roberta base         CV 0.4758   public LB 0.468  
- F23  roberta large         CV 0.4792   public LB 0.465  
- F32  deberta large        CV 0.4749   public LB 0.469  
- F41   roberta large mnli CV 0.4796   public LB 0.475  
- F50  funnel large           CV 0.480     public LB 0.465  
- F51   electra large          CV 0.479     public LB 0.469  

Models with same name have different parameters, you can check respective notebook for details.

I use one one drop based on CV to select models, although my best single model deberta large with CV 0.4757 public LB 0.462 private LB 0.462, it is not included in my final inference notebook.

**Ensemble**
I use same CV split across all models and ensemble performance is evaluated on oof predictions.
- Ridge regression CV 0.4553 public LB 0.451 private LB 0.449
- Although manual features didn't help when I integrate them with predictions of transformers by a linear model, it helps when I use lightgbm. And average predictions of lightgbm and ridge gives my final score： CV 0.4532 public LB 0.454 private LB 0.448. As public score is worse so this might not a real improvement.

**Code**
- ITPT
[https://www.kaggle.com/w5833946/f1-itpt](url)
- Train
[https://www.kaggle.com/w5833946/f2-train](url)
[https://www.kaggle.com/w5833946/f17-train/data](url)
[https://www.kaggle.com/w5833946/f23-train](url)
[https://www.kaggle.com/w5833946/f32-train](url)
[https://www.kaggle.com/w5833946/f41-train](url)
[https://www.kaggle.com/w5833946/f50-train](url)
[https://www.kaggle.com/w5833946/f51-train](url)
- Inference trainning set
[https://www.kaggle.com/w5833946/f-inference-1](url)
[https://www.kaggle.com/w5833946/f-inference-1-1/data](url)
- Inference
[https://www.kaggle.com/w5833946/f-inference-3-1-v4](url)

You need to remove some private models that loaded (but not used when inference) to re-run some of the noetbooks.
