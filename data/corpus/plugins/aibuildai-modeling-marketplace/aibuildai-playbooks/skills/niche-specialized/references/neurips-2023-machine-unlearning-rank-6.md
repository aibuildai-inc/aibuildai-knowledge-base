# 5th place solution

Competition: neurips-2023-machine-unlearning
Rank: #6
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/458531

Firstly, thank you for hosting this intriguing research competition! I'd also like to express my gratitude to everyone for sharing numerous important insights in discussions.

We have published our notebook, and the description is as follows:
Notebook : https://www.kaggle.com/code/marvelworld/toshi-k-rotate-and-marvel-pseudo-blend48-52
(The code for adding to the dataset and debugging does not affect the submission in any way)


## 1. Overview

Our solution is an ensemble of two approaches: (1) Retraining from transposed weights and (2) Fine-tuning with pseudo-labels.

* (1) involves retraining the model using a modified version of the original model. In this modification, all weights in Conv2D are transposed. This process helps in forgetting samples in the forget-set, enabling the reuse of valuable features from the original model.
* (2) reproduces the behavior of the retrained model with pseudo-labels. We estimate the mistakes made by the retrained model with the forget data and use them as pseudo-labels for fine-tuning the pretrained model.

One of these approaches is executed each time unlearning is performed for one model.


## 2. Retraining from transposed weights (by @toshik)

At the start of the unlearning process, all weights in Conv2D are transposed both vertically and horizontally. This operation facilitates the forgetting of the forget-set. The modification is carried out simply as follows,

```
def rotate_weight(local_model):
    print('rotate weight')
    for module in local_model.modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            module.weight = torch.nn.Parameter(module.weight.swapaxes(2, 3))
```
Afterwards, the model is trained with the retain set for 3 epochs. This is considered equivalent to inputting flipped images into the original model and performing fine-tuning.
When compared to retraining from scratch, transposed weights still retain valuable features from the original model. This allows for the reuse of such information, resulting in faster and more stable convergence during fine-tuning.


## 3. Fine-tune with pseudo-labels (by @marvelworld)

The first step is to infer the forget data from pretrained model. Next, infer the forget data after a simple unlearning as shown in the example notebook again. Compare the results of both inferences and identify the data on which the inference moves in the wrong direction.
We also developed a simple scratch model from retrain data to infer forget data. This result also identifies data on which the model is easy to mistake with some confidence.
Finally, finetuning pretrained model with the identified data and incorrect inferences as pseudo-labels.


## 4. Ensemble

Our solution comprises a mixture of two types of models. We opted for different combinations, and the final scores are as follows,

| Retraining from transposed weights | Fine-tune with pseudo-labels | Private LB |
| ---- | ---- | ---- |
| 246 models | 266 models | 0.0785184178 |
| 266 models | 246 models | 0.0756313425 |


## 5. What we tried but didn’t work

* class weights
* approach to each layer like LLRD,  init/freeze
* softmax labels / KL Loss
* Relax Loss
    * https://openreview.net/forum?id=FEDfGWVZYIn
* SCRUB
    * https://arxiv.org/abs/2302.09880

Although Relax Loss and SCRUB appeared to be a state-of-the-art methods in this field, they unexpectedly did not perform well.
The unlearning metric was so stringent that the defenses against MIA were insufficient to achieve a good score.

## 6. Final Remarks

Our solution is built upon two distinctive approaches, contributing to the stability of our solution in the private leaderboard. However, there are certain limitations when applying it to real-world problems, particularly in manually tuning some parameters.

We hope that our solution serves as a valuable reference in future research.
Thank you for reading this to the end!
