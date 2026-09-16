# 30 th Place Simple Solution & Eid Mubarak

Competition: nbme-score-clinical-patient-notes
Rank: #30
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322994

First of all, Eid Mubarak [A common greeting during Eid al-Fitr (the greatest festival of Muslims) is "Eid Mubarak," which means "Blessed feast, festival, or celebration".]

It is a great day for me as I celebrated **two Eid** at a time. One is Eid al-Fitr and another is achieving my **first solo silver** in any Kaggle Competition 

I would like to thank the organizers for hosting such nice competition. In this competition my goal was to learn from the *Kagglers*. So, I followed the public notebooks. My solution is based on the notebooks.

# Training 
For training I follow two public notebook. The top voted [notebook](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train) by @yasufuminakama and another awesome [notebook](https://www.kaggle.com/code/librauee/train-deberta-v3-large-baseline) by @librauee 

# MLM 
For MLM I use [this amazing notebook](https://www.kaggle.com/code/nbroad/deberta-mlm-tests-nbme). I keep the configuration unchanged but create two version of it with different eval data.

# Model 
Basically I focused on only two models as backbone 
1. DeBerta-v1-base : For any kinds of experiment 
2. DeBerta-v3-large : As a Final Model

# Different Classification Head
Due to choosing only one model for submission, to bring diversity (small) I use different classification head
1. Linear Head
2. Last 4 layers sum
3. Last 4 layers concat (after projecting those layer outputs)
4. LSTM Head
5. 1d CNN Head
6. 2 layer MLP Head

# Post Process 
I used exactly same post-process techniques that used in [this notebook](https://www.kaggle.com/code/theoviel/roberta-strikes-back) by @theoviel 
It improves LB by +0.003

# Ensemble 
For ensemble I use [this notebook](https://www.kaggle.com/code/motloch/nbme-ensemble-debertas) by @motloch 

# Some Tricks 
1. During training time, I use some tricks like AWP that are discussed [in this thread](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707) by @hengck23 
2. I use 5 fold CV. I observe CV/LB score for each fold. For the final ensemble, instead of making prediction of all folds of each models, I focus on different folds on different models which have good CV/LB correlation
3. For ensemble more models I used the padding optimization trick used [in this notebook](https://www.kaggle.com/code/anyai28/fast-inference-by-padding-optimization) by @anyai28 

# Things I learn 
1. Experiment is all you need
2. Following the Discussions and Notebooks in a competition 
3. Trust the CV 
4. Exploring Competition Related paper 
5. Try to avoid training using Kaggle GPU (but I have no option)

I want to give a special thank to @abdulkadirguner for suggesting me to submit a model to the competition when I have only few seconds GPU time [in this thread](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/322593)
