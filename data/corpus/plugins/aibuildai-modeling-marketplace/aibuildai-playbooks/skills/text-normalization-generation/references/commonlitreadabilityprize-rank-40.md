# 40th Place solution: How we improved our rank from 1100th three days ago

Competition: commonlitreadabilityprize
Rank: #40
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258363

First of all @shahules and I would like to start by heartily thanking the following people: @rhtsingh, @maunish, @jcesquiveld, @andretugan, @jacob34, @kurupical, @cdeotte and @markwijkhuizen. Without their great work and discussion it would not have been possible for us to have made it here. Additionally I would like to  personally thank @radek1 for his wonderful book **Meta learning**. This book inspired me with a lot of ideas in general as well as I used them in this competition. I highly recommend this book to all fellow data scientists. This is my first serious Kaggle competition. I did one for a couple of days five years ago but didn't pursue it. But this time I survived :)  Once again a very big hearty thank you to all of you. I personally had never worked with transformers two months ago. Thanks to this competition I am happy I discovered it. We were at 1100th place three days ago, so we are really happy that we made it to 40th.

Getting inspired from @rhtsingh and @maunish I decided to develop everything publicly for the first two months of the competition. Later on @shahules and I met exactly one month ago and we formed a team.  

We understood that the key to succeed in this competition were the following:
1. Stabilizing the training of the transformer
2. Leveraging massive external data
3. A good ensemble of good models
4. Definitely not to overfit

### Stabilizing the training of the transformer
Of all these things the most difficult part was stabilizing the training. Our validation graphs looked like seismographs. Here is a snippet of the graph before and after stabilizing.


At least with the base model it was ok but with the large model it was just impossible. We also felt very uncomfortable with evaluating every n iterations. Therefore we spent about twenty days focusing  on running experiments to stabilize the training. That's when we came across @jcesquiveld's models trained using **differential learning rate**. The first time when we implemented it felt magical. The training was extremely stable, no more seismographs ;) This helped us in avoiding frequent evaluation and also let us comfortably train our models on the full dataset without validation. Yes the idea of training on full data instead of Kfold CV came from @cdeotte. You can watch the wonderful video posted by @abhishek in his youtube channel where 4 grandmasters share their secrets of success


### Design of work
We also realized a good ensemble was key to success and independence is the key to successful ensemble. Therefore we always designed and ran experiments independently and discussed results later on so we don't bias each other as well.  In order to facilitate identical experimental conditions and also not to do double work we created a central repository of all our pipeline components adhering to good coding principles.  

### Choosing the models and not overfitting
 As the CV and LB correlation was not great, we had two choices: we could either probe LB and understand more about the test data but we had almost no time/submissions left for that. Our other choice was choosing models not based on their CV scores but by looking at the validation graphs, evaluating them visually if the training and validation loss look stable, and then if the final model was close to the best model (with frequent evaluation) and so on. So this was more of the data scientist instinct that helped us. We never chose a model because the CV score was high.  

### Model definition
The attention head that were used in almost all the public notebook we came across took the pad tokens into account. Therefore we modified it to exclude the pad tokens in the attention calculation. 

``` python
class MaskAddedAttentionHead(nn.Module):
    def __init__(self, input_dim, head_hidden_dim):
        super(MaskAddedAttentionHead, self).__init__()
        self.W = nn.Linear(input_dim, head_hidden_dim)
        self.V = nn.Linear(head_hidden_dim, 1)
        
    def forward(self, x, attention_mask):
        attention_scores = self.V(torch.tanh(self.W(x)))
        attention_scores = attention_scores + attention_mask
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * x
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x
```

Once we stabilized the training, replacing RMSE loss with MSE loss gave us a good boost. We didn't dig deeper into the reason but I guess it had something to do with gradient flow. 

### Leveraging massive external data
One of our star differences was the pretraining of our models with the external data. Thanks to @markwijkhuizen we did not have to scrape Wikipedia. We converted the problem into a text ranking problem and trained the model with the ranking loss as the objective. We understood that Bradley Terry scores were essentially computed by comparing pairs of text excerpts therefore we have to do something similar. As many of the text excerpts come from Wikipedia and Simple Wikipedia we trained a ranker as follows: 
Our initial experiments with pretraining did not give us good results so we dropped it. Only on the last day after running out of ideas we remembered abandoning this and thought why not try it again.  The reason why transfer learning did not work well in the past for us but worked well now was because our transformer became a stable bed now. 

Now when we look back the key to rapid progress in three days comes from the fact that we invested all our effort in preparing a very stable experimental set-up. 

Thank you all for all your valuable discussion in the forums and notebooks. Without great work being made public, we couldn't have done this. I really cannot thank @maunish and @rhtsingh enough. This continues to inspire us to develop more and more publicly. Thank you.

Here is our code: https://www.kaggle.com/vigneshbaskaran/commonlit-nn-kit
