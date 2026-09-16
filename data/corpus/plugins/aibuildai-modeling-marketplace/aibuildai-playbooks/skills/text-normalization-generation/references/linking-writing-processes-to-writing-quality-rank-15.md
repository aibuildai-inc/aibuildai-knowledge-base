# 15th solution feature selection and trust your CV

Competition: linking-writing-processes-to-writing-quality
Rank: #15
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/467674

Congratulations to everyone who won or learnt from this competition.

Unfortunately, we did end with a gold medal, but we have already selected our best private result. Our best result come from feature selection of two groups of features & models. Our strategy is feature selection + trust CV

**Features summary:**
- P1 (features 1, 2, 3): from my teammates, but unfortunately they’re busy on their own work, they can not write up the detail.

- P2: from my sides, which has 4 subgroups of features, feature4, feature5, feature6 and feature8 with feature selection, feature select boosted the CV about 0.008, but worsen 0.003 on public LB.

**Feature 4/5/6/8 details:**
- Feature4: the public 0.584 notebook, I did feature selection and keep 130 features
- Feature5 (selected 45 features): my own features but most are borrowed from public notebooks, and I added below features 
TFIDF of constructed essay: 

TFIDF of event: 

‘word_comma_cusor’ feature: 

- Feature6 (selected 60 features): added burst features to feature5 

- Feature8: public 0.582/581 features, with feature selection of top 80 features


**Final result:**
- 0.5*my_teammate_result(CV: 0.587, public LB: 0.576, private LB: 0.566) + 0.5*my_result(CV: est 0.590, public LB: 0.579, private LB: 0.568)

- my_results = 0.4*freature4_130features + 0.125*feature4_45features + 0.175*feature6_60features + 0.3*feature8_80features

- feature5 and feature6 used 5 Kfold split, LGB, CAT, XGB and SVR with Bayesian optimisation to select best weights.

- Feature4/5/6/8 used 5 seeds. feature4 CV: ~0.6, public LB 0.587; feature5 CV: 0.592; public LB: 0.590, feature6 CV: 0.591, public LB: 0.588; feature8 CV: 0.608, public LB: 0.584. Ensemble of feature4/5/6/8, got public LB: 0.579, private LB: 0.568.
