# 18th place solution - You really don't need big models

Competition: shopee-product-matching
Rank: #18
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/237972

[Edit] I have updated my code to incorporate the idea of Neighborhood Blending shared in the [1st place thread](https://www.kaggle.com/c/shopee-product-matching/discussion/238136) it now scores 0.762/0.772

A thing I noticed in discussions and kernels throughout this competition was that people focus too much on models, architectures and  training. And not enough on how to actually solve the problem - turn model outputs into label groups.

To prove the point I used my final submission to replace models in my code with the following:
**resnet18+bert+tfidf**
It scored .749 putting it comfortably in the silver medal position.

* The combined training time of the models was under 30 minutes (on a 3090 GPU).
* I've made a single submit with it, so no special parameters tuning.
* I trained a single fold, so only used 80% of the train data.

The kernel is here:  https://www.kaggle.com/slawekbiel/resnet18-0-772-public-lb


I'm not saying that bigger models are not useful at all (my best solution scored .767 with nfnet_l0+nfnet_l1+ Bert-large+tfidf) but the biggest gains I've seen  were from changes in the code, not the models.

# The outline of my approach

### Step 1 training 
I used arcface for all my training, and fine tune pretrained models with it for a few epochs. For image I tried resnet, efficient net and nfnet. The nfnet worked the best for me so that’s in my final solution. The resnet18 I used in the toy solution just to show a very basic model can work too. 

For text I found that language model pretrained on Indonesian language worked the best.

### Step 2 Embeddings and similarities 
For each model I first generate embedddings and then calculate the full cosine similarity matrix.

### Step 3 Combine the model outputs
I combine the matrices from the previous step with formula D = 1 - (1 - D_1) * (1 - D2) * (1 - D3)

This I found works much better than alternatives of taking mean or max.

### Step 4 Rerank
I replace the predictions of each row with an average of predictions of its nearest neighbors. I set threshold for “nearest” so that 4 neighbors are used on average (found experimentally)

It helps the score a bit.

### Step 5 force groups into a desired distribution 
This was a largest single change I made, jumping my score from .739 to .759
I nicknamed it “chiseling” the idea is: I first make an educated guess that the distribution of group targets in the test data is similar to the one in train. Then I make my solution to have the same shape.

I first decide that groups with 2 elements are going to be those where the third largest element is the lowest. Then I follow the same logic for all the sizes up to 50.

### No thresholds!
Early in the competition I followed the public approaches of selecting elements to pick based on the hard coded threshold - this had to be tuned for test data separately and eat into your submissions limit and time. (Especially with multiple models). When I moved to make decisions based on the data distribution instead, it made the code much more resilient with less need for tuning when changing/adding models.
