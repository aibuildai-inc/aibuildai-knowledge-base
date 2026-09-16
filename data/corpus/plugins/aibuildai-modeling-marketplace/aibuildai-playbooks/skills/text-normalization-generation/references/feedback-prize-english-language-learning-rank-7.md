# 7th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #7
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369736

Congrats to all the winners, and thanks to hosts for interesting competition.
I still can't believe I shaked up to gold zone.

# overview
My final submission is ensemble of 16 models. The weights of the models are determined using nelder-mead.


# model

The default parameters are as follows:

- max_length = 512
- awp 
  - start_epoch = 3
  - adv_lr = 0.0001
  - adv_eps = 0.01
- bce
- lr = 2e-5
- lwld
  - base model: lr_decay = 0.9 
  - large model: lr_decay = 0.95 
- freeze layers 
  - large model: 1/2
  - xlarge model: 3/4
- fp16
- 5-fold


| model | changed | cv |
| ---- | ---- | ---- |
| deberta-v3-base | max_len=1024 | 0.4517
| deberta-v3-base | no awp | 0.4572
| deberta-v3-large |  | 0.4525
| deberta-v3-large | no freeze | 0.4549
| deberta-v3-large | max_len=1024 | 0.4566
| deberta-v3-large | l1_loss | 0.4576
| deberta-v2-xlarge | lr=5e-6| 0.4602
| deberta-xlarge |  | 0.4514
| deberta-large |  | 0.4502
| deberta-large | no awp | 0.4555
| deberta-large | l1_loss | 0.4557
| deberta-base |  | 0.4567
| roberta-large |  | 0.459
| muppet-roberta-large | no awp | 0.4635
| distilbart-mnli-12-9 |  | 0.4626
| bart-large-finetuned-squadv1 |  | 0.4635

# What worked
- [High Impact] awp
    - cv+0.004 ~ 0.005
    - awp didn't work for public lb, so I added no awp model for ensemble

- [Middle Impact] lwld
   - cv+0.002~0.003

- [Middle Impact] nelder-mead method

- [Low Impact] bce
    - converged faster than l1_loss

- [Low Impact] freeze layers
    - stable and faster training

# What Didn’t Work
- fgm
- Last Layer Re-initialization
- lgb stacking

# Important Citation
- feedback3
  - [FB3 / Deberta-v3-base baseline [train]](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)

- feedback1
  - [feedback1 1st solution](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)

- commonlit
  - [commonlit 2nd place solution](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328)
  - [The Magic of No Dropout](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260729)

# inference code
https://www.kaggle.com/code/tanakar/fb3-ensemble-per-col/notebook

# Thanks and Acknowledgements:
Thanks to our hosts for hosting the competition. And thanks to the Kaggler's for sharing their helpful notebooks and discussions.

# Team Members:
- [@tanakar](https://www.kaggle.com/tanakar)
