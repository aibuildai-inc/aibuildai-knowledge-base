# Private 2nd/Public 5th solution: Py-Boost and GCN

Competition: cafa-5-protein-function-prediction
Rank: #2
Source: https://www.kaggle.com/c/cafa-5-protein-function-prediction/discussion/434064

Hi everyone!

First, thanks to Kaggle and CAFA teams for the competition. The task we were solving was both important and challenging and even though there were some frustrating moments such as low protein amount in the LB and unobvious evaluation code, we enjoyed a lot while solving this problem.

Here is a brief overview of public 5th place solution:

### 1) Sequence embedding
We tried to use the following embedding list: T5, esm2-large, ankh-large. Finally, the most of the models used only T5, but some of them used concat of T5+ESM. In addition, we concatenated one-hot taxon features to embedding. We selected only taxons that are good enough represented in both train and test features, around 30 totally, other taxons where merged into single group.
### 2) Base models
##### Validation
We were not able to create a validation scheme better than a simple 5 fold CV. We had some experiments on the topic but other CV schemes lead to the models with less LB score.
##### Py-boost
The best performed models on both CV and LB are from Gradient Boosting family. We used my own GBDT implementation called `py-boost`. I made it few years ago especially to deal with extreme multi-output datasets since it was my main research area at that time. It works on GPU only and it is able to train multi-label tens on even hundreds times faster than  popular well known implementations. You can check `py-boost` on the [github](https://github.com/sb-ai-lab/Py-Boost) or read our NeurIPS [paper](https://arxiv.org/pdf/2211.12858.pdf) where we explain all the strategies to speed-up multi-output training.

Locally I was able to fit 4.5к output (3000/1000/500) `py-boost` on a single V100 32GB GPU and it takes about 1.5 hour for a single fold.

##### Logistic Regression
We also trained a simple 13к output Logistic Regression. It shows much less performance than GBDT on the popular terms, but is able to perform on rare outputs.

##### Neural networks
In the ensemble of models, we used a slightly modified version of the public notebook [Pytorch,Keras,Etc 3 Blend, CAFA metric, etc](https://www.kaggle.com/code/alexandervc/pytorch-keras-etc-3-blend-cafa-metric-etc#Optimizer-%22Sophia%22-sometimes-better-than-Adam). The only difference was the best cross-validated combination of hyperparameters averaged many times.

### 3) Alternative modelling approach - predicting the conditional probabilities
This approach was discovered at the beginning of the competition thats why it makes some wrong assumptions about the data. But somehow it becomes useful for us for both CV and LB. The main advantage of this approach is utilising the OBO graph on the inference phase and it helps to make a prediction even for terms that were not used in training. Here are the main points:
- We assume, that term can exist for the protein only **if at least one of its parents exist**. This is wrong. In real, if term exists, all its parents exist too because of the propagation rules.
- We reformulate classic multi-label scheme where target matrix of shape (n_protein, n_terms) consists of 0 and 1 to the new scheme. Now targets can be 0, 1 and NaN. Term for the protein will have NaN value in matrix if there is no parent term with value 1. 
- During model training phase NaN cells in the target matrix are masked and ignored
- Now, our model outputs **the conditional probabilities of term in case of at least one of its parents exist**. On the inference phase we need to transform it back to the raw probabilities
- Transformation is made in the order defined by graph. When we process the term, all its parents are already processed and have raw probabilities. All terms are included to the scheme, even they are not used in training. For the terms that were not used for training, we used prior mean.
- While processing the term, we make another wrong assumption, that parents probabilities for the term are independent. But if we assume that, according to [1], we can calculate raw probability for term as
	 `p_term_raw = p_term_cond * (1 - (1 - p_parent_0_raw) * (1 - p_parent_1_raw) * ... * (1 - p_parent_N_raw))`. 
	 Remember that while we processing the term, all its parents already have raw probabilities calculated.

That approach scores better on both CV and LB, but not to much. The main advantage of it is that models are very different from classic multi-label approach and combining it all together boosts our score a lot. We apply this technique for GBDTs and LogRegs, so finally we had 2 GBDTs, 2 LogRegs and single blended NN as the base models

### 4) Stacking with GCN

We used graph convolution network to aggregate all the predictions. It is trained for the node classification task where each node is a term and each protein is a graph (but all the proteins have the same adjastency matrix). As the node features we used base models predictions together with node embedding trained from scratch. We also added GO annotations features described in the next section.

One interesting feature we discovered by just making a mistake. On the inference stage we have wrong model ids but still get a good score, so we made a kind of test time augmentations by averaging the predictions with shuffled models.

Another important thing I should mention is about the metric. As far as I understand, we are evaluated only on protein/ontology pairs that are experimentally found. So, if ones we predict a term from ontology that does not exist, we will not get any penalty at all! That means, that we need actually to estimate **the conditional probability of term in case when its ontology exists**.  So the correct way to fit a final stacker model for ontology X is to truncate the sample and take only the proteins that contain the terms from ontology X. It gives some small boost to the score and speed-up the computations. 

### 5) GO annotations

We discovered the GO annotations dataset provided by the [link](http://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz). They provide not only labelling but also the evidence codes of each term. We separated the codes that Kaggle suppose to be experimental (hope we understood it correctly) from the electronic codes. So we can use electronic labelling as the features to predict experimental given by Kaggle. From our analysis we discovered that about 30% of electronic labels becomes experimental, so using it as model feature performs better than just adding it as is. Experimental labelling was added as is for about 500 proteins we were able to find in this dataset. We also added raw labelling that MT provided if we can not observe it in our dataset. The last step almost didn't change our score, so that sources are almost the same. 

### 6) Postprocessing

We also used the OBO graph to make a post processing. The problem of ML model prediction of protein terms is that it is inconsistent. Following to propagation rule that is applied to the target (I checked it in the CAFA-Evaluator [repo](https://github.com/BioComputingUP/CAFA-evaluator)) if term exists, all its parents assumed to exist too. So, consistent model will never predict the probability for parent lower than term probability. But our models don't care about it at all. So we can manually fix this situation. Our final term prediction is **the average of term probability, maximum propagated children probability, and minimum propagated parents probability**. That trick boosts score just a little on the LB but hope it makes model a little bit more stable.

### P.S.

We wish a good luck for all the competitors since of course as many of you we expect a big shake up this time. I hope all our work will not be wasted :). Looking forward to see the final results

**Update** All the solution code is open source now and available at [Github](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place)
