# 27th Place Solution: How to use prompts?

Competition: commonlit-evaluate-student-summaries
Rank: #27
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446542

Firstly, I want to  thank  Kaggle and the Host for organizing this fun competition, and thanks to my teammates @muhammad4hmed , @mohammad2012191 , @cody11null , @ivanisaev . This wouldn't be possible to achieve without them!
# Summary
Our solution is an ensemble of diverse 1st stage models (Mostly with Deberta-v3-large backbone) feed into a 2nd Stage stacking Model. Diversity came from input, different custom pooling techniques and heads.
Our best submission out of the 3 we chose had CV 0.486 and LB 0.425. It gives PB 0.464 which puts us at 27th. The best submission out of all our submissions had LB of 0.459, so it is a silver in both cases :)
# First Stage Models
We finetuned Deberta Base, Large and Squad using summary texts + question only. We also added  other models with some custom pooling ideas which involved using prompts inside, and they worked pretty nice, thanks to @ihebch and @muhammad4hmed work! 
### Model 1

CV: 0.505; public LB: 0.456; private LB: 0.474 
### Model 2 (not included in ensemble 😢 )
same as Model1 but without cross-attention mechanism
CV: 0.49; publicLB: 0.575; privateLB: 0.468 (wow)
### Model 3

CV: 0.52; PublicLB: 0.457; PrivateLB: 0.499
# 2nd Stage Model
We then added all these predictions as features for our LGBM+XG+CB models along with the same features published in public notebooks. 
We also added **FB3** labels (good boost) and used metadata features(specially grade).
We tried hard to add new features/ do feature selection / do extensive hyperparameters tuning, but all these trials ended up with better cv and much worse lb, so we gave them up and focused on improving our 1st stage models.
# What Didn't Work
- Feature Selection
- Extensive Hyperparameters Tuning
- Adding FB2 Labels
# Team Members
- @ivanisaev 
- @cody11null 
- @mohammad2012191
- @muhammad4hmed
- @ihebch
