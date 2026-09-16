# [Private 28th/public 14th solution]-Tri's part-custom loss

Competition: commonlitreadabilityprize
Rank: #28
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258312

First of all, I would like to say thank you very much to my wonderful teammates, @horsek, and @lftuwujie, I learned a lot from them during this competition. Next, thank you to all competitors, the competition host, and Kaggle for competing and organizing this amazing competition. Besides, congratulation to all teams in the gold medal zone.

At the beginning of this competition, I told @horsek that let's treat it as our chance to learn about NLP because I didn't have knowledge of it and it has been my first NLP competition.

In this write-up, I would like to share not only my models but also my journey in this competition, from starting points to forming ideas under a newbie's viewpoint like me.

# Starting point
I started my journey with this wonderful notebook (https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-fit), please give him/her an upvote.

However, I didn't fork it but read and re-wrote my own notebook. This helped me to debug, and forming some ideas to further customize and optimize the pipeline.

After this step, my best single model is the RoBERTa-large and scored 0.468 on the public LB and 0.469 on the private LB, my inference notebook is [here](https://www.kaggle.com/shinomoriaoshi/readability-roberta-large-inference?scriptVersionId=65661288)

The setting of this model is,
* Epochs: 8;
* Batch size: 8;
* LR: 2e-5;
* Evaluate after every 2 steps;
* Differentiate learning rate of hidden layers;
* Model: Backbone --> Weighted-mean/max pooling of hidden layers --> Attention pooling over document length --> Output
* Loss: RMSE
* Optimizer and scheduler: AdamW, get_linear_schedule_with_warmup

# Custom loss
In this competition (and also in all of my competitions), I studied related literature a lot. This step helped me to understand the concept of the competition and find new ideas. The first idea I got is the Quadratic-Weighted-Kappa loss, the second is the custom Bradley-Terry loss.

### Quadratic-Weighted-Kappa (QWK) Loss
The idea of this loss came from this [paper](https://direct.mit.edu/coli/article/47/1/141/97334/Supervised-and-Unsupervised-Neural-Approaches-to). The main idea of this loss is binning the excerpt targets into categories, and train the model to classify them into these categories. In my opinion, this step is equivalent to label-smoothing, because it gathers excerpts with similar targets into a bin. The QWK loss is not a must, you can implement it with Cross-entropy loss. This is the implementation of this QWK loss,

```
class QuadraticWeightedKappaLoss(nn.Module):
    def __init__(self, num_cat = 7, device = 'cpu'):
        super(QuadraticWeightedKappaLoss, self).__init__()
        self.num_cat = num_cat
        cats = torch.arange(num_cat).to(device)
        self.weights = (cats.view(-1,1) - cats.view(1,-1)).pow(2) / (num_cat - 1)**2
        
    def _confusion_matrix(self, pred_cat, true_cat):
        confusion_matrix = torch.zeros((self.num_cat, self.num_cat)).to(pred_cat.device)
        for t, p in zip(true_cat.view(-1), pred_cat.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        return confusion_matrix
        
    def forward(self, pred_cat, true_cat):
        # Confusion matrix
        O = self._confusion_matrix(pred_cat, true_cat)
        
        # Count elements in each category
        true_hist = torch.bincount(true_cat, minlength = self.num_cat)
        pred_hist = torch.bincount(pred_cat, minlength = self.num_cat)
        
        # Expected values
        E = torch.outer(true_hist, pred_hist)
        
        # Normlization
        O = O / torch.sum(O)
        E = E / torch.sum(E)
        
        # Weighted Kappa
        numerator = torch.sum(self.weights * O)
        denominator = torch.sum(self.weights * E)
        
        return numerator / denominator
```

My best model after this step is again a RoBERTa-large with the following setting,
* Epochs: 8;
* Batch size: 8;
* LR: 2e-5;
* Evaluate after every 2 steps;
* Differentiate learning rate of hidden layers;
* Model: Backbone --> Weighted-mean/max pooling of hidden layers --> Attention pooling over document length --> Output
* Loss: RMSE + QWK
* Optimizer and scheduler: AdamW, get_linear_schedule_with_warmup

It scored 0.466/0.466 on the public/private LB.

### Bradley-Terry (BT) Loss
This is so far the most effective trick and my most contribution to my team. The idea came from the observation about the benchmark excerpt which its target and standard error are both 0. As I studied the information from the host, each excerpt in the corpus is compared to some others. Then, by using the Bradley-Terry method, their prior targets are estimated. Finally, the prior targets of all excerpts are subtracted to the target a chosen benchmark excerpt to obtain the data we have.

From this observation, I designed my models accordingly. The model architecture is [here](https://www.kaggle.com/shinomoriaoshi/readability-model-architecture) (sorry that I couldn't insert an image to a post, so I uploaded my model architecture in another notebook).

The implementation of the custom BT loss is as follow,
```
class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super(BradleyTerryLoss, self).__init__()
        
    def forward(self, pred_mean, true_mean):
        batch_size = len(pred_mean)
        true_comparison = true_mean.view(-1,1) - true_mean.view(1,-1)
        pred_comparison = pred_mean.view(-1,1) - pred_mean.view(1,-1)
        return torch.log(1 + torch.tril(torch.exp(-true_comparison * pred_comparison))).sum() / (batch_size * (batch_size - 1) / 2)
```

With this structure, the predicted target of the benchmark excerpt is always 0, which is consistent with the truth. This loss also allows models to compare the targets to each other. We applied this custom loss with many models of my teams.

This custom loss gave me the best single model so far with 0.461/0.462 on the public and private LB.
My best model after this step is again a RoBERTa-large with the following setting,
* Epochs: 20, stop after 8 epochs;
* Batch size: 8;
* LR: 2e-5;
* Evaluate after every 10 steps;
* Differentiate learning rate of hidden layers;
* Model: Backbone --> Weighted-mean/max pooling of hidden layers --> Attention pooling over document length --> Output
* Loss: RMSE + QWK + BT
* Optimizer and scheduler: AdamW, get_linear_schedule_with_warmup

Using this setting, I train my models with different types of backbone. The following summarizes my best models according to public LB with each backbone (public LB/private LB),
* RoBERTa-large with weighted-mean/max pooling: 0.461/0.462;
* RoBERTa-large with max-pooling only: 0.463/0.466;
* GPT2-medium: 0.492/0.508;
* XLNet-large-cased: 0.468/0.473;
* ALBERT-xlarge-v2: 0.502/0.500;
* ELECTRA-large: 0.468/0.476;
* Funnel-transformer/large: 0.477/0.477;
* BART-large: 0.487/0.491;
* DeBERTa-large: 0.463/0.468;

For each backbone, I sometimes changed the setting (evaluate less frequently, omit the QWK loss, training with another fold splitting, etc.) to avoid overfitting.

Finally, I simply took the equal average of all of my 13 models, it gave me 0.453/0.456 on the public/private LB. Ensemble my average model with my teammates' models gave us 0.448/0.452 on the public/private LB.

# What didn't work for me?
* Data augmentation;
* Linguistic features;
* HAN

# Final words
Although the result is not as good as I expected, I still enjoyed the competition a lot. From nearly 0 in NLP, after 3 months, I learned a lot from my teammates and fellow competitors. I hope to meet you in future competitions!

# Notebooks
* Training of my models: https://www.kaggle.com/shinomoriaoshi/readability-training-notebook
* Inference of my stacking model: https://www.kaggle.com/shinomoriaoshi/readability-stacking-inference
* OOF notebook of a RoBERTa-large model: https://www.kaggle.com/shinomoriaoshi/readability-roberta-large-oof
