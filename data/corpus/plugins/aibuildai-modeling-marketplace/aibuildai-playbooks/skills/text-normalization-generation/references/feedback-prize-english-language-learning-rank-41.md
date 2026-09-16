# 42th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #41
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/371203

First of all, thank you for hosting this competition.
I learned a lot about the NLP of this competition.
Congrats to the winners!

# Overview
- I use 11 models(4 fold model x 4 and 10 fold model x 7).
  - First, I use LGBM(with optuna) for 4fold model.
 ※ I didn't use 10 fold model because the experiment was not completed in time for the competition deadline..
  - Second,  I did weight ensemble. 
- cv=0.4426811/PublicLB=0.435475/PrivateLB=0.435906

- 10 fold model is open model([link1](https://www.kaggle.com/code/jingwora1/fb3-deberta-family-inference-weight-tune),[link2](https://www.kaggle.com/code/kojimar/fb3-single-pytorch-model-inference)). So I share only 4 fold model detail of cv score. 

| model | CV score |
| --- | --- |
| (*1,4,5)deberta-v3-base (pseud label fb1,2) | 0.4528044 |
| (*4,5)deberta-v3-large | 0.4568590 |
| (*2,4,5)deberta-v3-base (topics) | 0.4537212 |
| (*3,4,5)deberta-v3-base (original loss) | 0.4564600 |

※ (*1,4,5) use 「Pseud Labeling」,「Concat last Four hidden」and「Dropout」.

# What Worked
- Pseudo Labeling(*1)
   - I use all data of fb1 and fb2.
   - I use MultilabelStratifiedKFold to fold datasets.
- Add topic(*2)
   -  I refer [this code](https://www.kaggle.com/code/jdoesv/take2-feedback-essays-to-prompts) and predict topic.
   -  I add predicted topic to text. (ex. [1_students_online_school_classes]I think that~)
- oliginal loss(soft)(*3)
   -  Much of the data for this competition was concentrated around 3. So I changed the loss to be more robust to outlier value.
   -  sample is below.
```
self.mse = nn.MSELoss(reduction='none')
l = self.mse(y_pred, y_true) + eps # eps=1e-9
loss = torch.mul(torch.pow(l,alpha),torch.sqrt(l))
# soft→alpha=1, hard→alpha=2
```
- Concat last Four hidden(*4)
    - MLP model include important information in  last Four hidden layers.
    - I refer [this code](https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently) 
-  Dropout(*5)
    - Dropouts can reduce overlearning.
    -  sample is below.
```
self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
output = sum([self.fc(dropout(feature)) for dropout in self.dropouts])/5
```
- LGBM
    - I make 6LGBM model to predict 「cohesion」, 「syntax」, 「vocabulary」, 「phraseology」, 「grammar」, and 「conventions」.
    - I use 2stage stacking.

# What Didn’t Work
- MLM
    - I can't get good LB and CV score
    - I think MLM is useful when datasets is small. But This competition can use fb1,fb2 and fb3.So I think MLM does not need.
- Focal loss(hard)
    - LB score dropped. So I didn't use.

# Important ciations
- FB3 / Deberta-v3-base baseline [train] [link](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
- FB3 / Deberta-v3-base baseline [inference] [link](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference/notebook)
- FB3 Deberta Family Inference [weight tune] [link](https://www.kaggle.com/code/jingwora1/fb3-deberta-family-inference-weight-tune)
- FB3 single pytorch model [inference] [link](https://www.kaggle.com/code/kojimar/fb3-single-pytorch-model-inference)

# Thanks and Acknowledgements
This was almost my first time to participate in an MLP competition.
I think this competition was perfect for someone with no MLP experience because I learned a lot from the Discussion. 
We were also able to reaffirm the importance of trust CV. 
Again, thank you very much for a very enjoyable competition!
