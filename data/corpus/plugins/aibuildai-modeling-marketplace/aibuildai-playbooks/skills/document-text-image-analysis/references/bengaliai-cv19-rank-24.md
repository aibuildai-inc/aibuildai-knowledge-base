# 24th place solution - Post processing(+0.013 Private LB)

Competition: bengaliai-cv19
Rank: #24
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136064

First of all, I would like to thank the organizers for holding a very fun competition. Thank you to all kagglers! Through the discussion, I learned a lot.


My solution used a simple ensemble model and post-processing to maximize macro recall. Post-processing adds a bias to the logit output by the model. Without post-processing I am out of medal zone😭


Here is an overview of the solution:

## Inference Label
First,  the DL model calculates the logits. As a post-process, add optimal bias to logits. Infer estimated labels using argmax for logits.




## Model
The model is an ensemble of five models. All training data was used for learning. Due to my limited time and computational resources, I've done very little cross-validation.

- model 1, 2, 3, 4  
  - Seresnext50_32x4d (pretrained model using imagenet)  
  - image size 128x128  
  - Three linear head (output is grapheme root logit (168), Vowel diacritic logit (168), Consonant diacritic logit (168))  
  - Mish activation, Drop block
  - Cross Entropy Loss  
  - AdaBound  
  - Change the following parameters for each model  
    - Manifold mixup (alpha, layer), ShiftScaleRotate, Cutout  
- model 5
  - Seresnext50_32x4d (pretrained model using imagenet)  
  - image size 137x236  
  - Three linear head (output is grapheme root logit (168), Vowel diacritic logit (168), Consonant diacritic logit (168))  
  - Mish activation, Drop block
  - Cross Entropy Loss  
  - AdaBound  
  - Manifold mixup (alpha, layer), ShiftScaleRotate, Cutout  


## Calculation of post-processing bias
Since the model is trained in Cross Entropy Loss, it generally does not maximize macro recall.  So I considered using a bias that would be added to logit to optimize macro recall. The optimal bias was calculated using a real coded genetic algorithm as follows:  



- Calculate logit for all training data.  
- Calculate bias to maximize macro recall with real-valued genetic algorithm.  
- Calculate logit for all training data with data extension.  
- Calculate bias to maximize macro recall with real-valued genetic algorithm.  
- The above biases are averaged to obtain the final bias.

In estimating the test data, the above bias calculated from the training data is used.




There is not much good English literature on real coded genetic algorithms. I think the following is helpful for implementation.
https://github.com/statsu1990/Real-coded-genetic-algorithm

CMA-ES is similar to real coded genetic algorithm, and there is a lot of English literature.
Therefore, you may want to try CMA-ES.  


## Score (Public LB / Private LB)
Single model  
- model1 (wo bias): 0.9689 / 0.9285  
- model2 (wo bias): 0.9680 / 0.9270  
- model3 (wo bias): 0.9691 / 0.9317  
- model4 (wo bias): 0.9681 / 0.9243  
- model5 (wo bias): 0.9705 / 0.9290  


Ensemble model
- ensemble1 ~ 5 (TTA, wo bias): 0.9712 / 0.9309  
- ensemble1 ~ 5 (TTA, with bias): 0.9744 / 0.9435


That's all. thank you for reading!
