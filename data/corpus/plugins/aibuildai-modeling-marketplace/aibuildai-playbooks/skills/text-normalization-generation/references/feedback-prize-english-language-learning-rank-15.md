# 17th place solution

Competition: feedback-prize-english-language-learning
Rank: #15
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369626

Thanks to Kaggle and the competition hosts. The Feedback competitions were a lot of fun to work on, and it is great to hear that more competitions will be launched soon. 

# Overview
My solution for this competition is an ensemble of 3 deberta models - one deberta-v3-base and two deberta-v3-large models. I relied on different training methods for diversity and ensembled them based on weighted average. For validation, I used MultilabelStratifiedKFold and my CV and LB were fairly correlated. Also, the essays in the training data are based on 42 topics and some topics had a lot of essays (100 - 300) while others had only a few essays (< 50). Models that performed well on the topics with fewer samples also performed well on the public LB, and on the private LB as well. 


# What Worked
- Training with AWP
    - Result: +.001
    - A lot of  top solutions in recent NLP competitions used AWP and it worked well for me in this competition. My strongest model was deberta-v3-large with AWP starting from the 2nd epoch. 


- Using differential learning rates
    - Result: .0008
    - Based on the discussion forums, this was one of the first things I tried and I ended up using it in all my models.

- Reinitialising the last layer
    - Result: Improved the ensemble CV by .001
    - I trained this model based on the suggestions in [this post](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/360819) and it worked well in the ensemble. 


# What Didn’t Work
- Pseudo labels 
    - I tried pretraining on pseudo labels, and using them while training as well but couldn’t get it to work. Probably needed to tune it some more as it seemed to have worked for other teams. 
- MLM
- Adding essay prompts while training

# Important citations:
AWP: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook


# Thanks and Acknowledgements
Thanks to everyone who shared training tips in the discussion forums, especially @wuwenmin. I joined a little late and reading all the discussions really helped.
