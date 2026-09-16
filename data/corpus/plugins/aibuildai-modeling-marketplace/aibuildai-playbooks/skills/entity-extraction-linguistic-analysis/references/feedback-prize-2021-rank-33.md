# 33rd Solution

Competition: feedback-prize-2021
Rank: #33
Source: https://www.kaggle.com/c/feedback-prize-2021/discussion/313331

First of all, thanks for competition host and competitor. In this competition I learned a lot about many insights and knowledge about NLP.
Especially I experienced NLP competition on Kaggle and token classification at the first time.

I modified my following baseline. Mainly I describe different and important points here
- training
https://www.kaggle.com/ytakayama/train-pytorch-longformer-5fold-forgooglecolab
- inference
https://www.kaggle.com/ytakayama/infer-pytorch-longformer-5fold

## training
- preprocess: replace LF to sep token.
This time line feed seems to be important so that we judge discourse type, because it might be the point discourse type changes.
- augmentation: replace 10% of tokens of each documents except special tokens to mask tokens
https://www.kaggle.com/spidermandance/masking-feedback-prize
- multi sample dropout: the same as the baseline
- model: BERT backbone + 15 class output(NER) the same as baseline

- optimizer: AdamW,
- weight_decay =0.01(except LayerNorm and bias)
-  lr scheduler: linear warmup(warmup rate:0.05)
- epoch: 6
- loss function: Cross Entropy Loss(label smoothing=0.1)
label smoothing is effective.
- max_length(inference): 1536
- cv: KFold(n_splits=5)
- batch size * gradient accumulation steps: fixed to 4.
I trained all models on single GPU V100/A100(Google Colab Pro+), to prevent from OOM I used gradient accumulation. I got A100 only when training deberta-large. Most of all batch size is 1 and accumulation step is 4

# detail about models
Including post process described at next section
|No |model |max length※ | max lr |cv|public LB| private LB|
| ---|--- | --- | --- |--- |--- |--- |
| 1 |allenai/longformer-large-4096  | 1536 | 1e-5| 0.6814| 0.683| 0.694| 
| 2 |funnel-transformer/large  | 1536 | 8e-6| 0.6926| 0.698| 0.709| 
| 3 |microsoft/deberta-large  | 1536 | 1e-5| 0.6956| 0.700| 0.712| 
| 4 |microsoft/deberta-v3-large  | 1024| 1e-5| 0.699| 0.703| 0.715| 
| 5 | 1+2+3   | -| -| 0.7059| 0.705| 0.717| 
| 6 | 1+2+3+4 simple average   | -| - | 0.7098| 0.709| 0.719| 
| 7 | 1+2+3+4 weighted average   | -| - | 0.7111| 0.710| 0.720| 


※at training
## post process
- fixed minimum length of words about each discourse type to 3
- Calculate minimum probability of each discourse type by using OOF so that f1 is higher and chose reasonable thresholds.
- fixed threshold of probability and tuned minimum length of words about each discourse type
- average probability of some models
No.7: No.1 * 0.1 + No.2* 0.25 + No.3 * 0.3 + No.4 * 0.35
- link evidence
https://www.kaggle.com/abhishek/two-longformers-are-better-than-1


## submission
I chose No.6 and No.7, best on both cv and lb 
Aggregated documents which are similar length to minibatch, I accelerated inference and colud ensemble 5 folds of 4 models.
To aggregate documents to minibatch, I calculate length of words instead of that of characters.
which is different from the following notebook.
https://www.kaggle.com/librauee/infer-fast-ensemble-models

## what did not work for me ( I cannot utilize better)
- symmmetric cross entropy loss
tuned alpha but convergence became worse 
- CRF
- use last some layers in hidden states
- add activation functions / arrange head of model
- calculate not first logits but average logits in each word


I wrote overview of my solution in a hurry. I may update it later.
In this competition I became competition expert and notebooks expert. Thanks for upvoting my notebooks and sharing great knowledge.
