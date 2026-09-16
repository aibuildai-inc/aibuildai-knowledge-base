# 4th Place (Public LB) Efforts

Competition: nfl-big-data-bowl-2020
Rank: #4
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119885

First of all, I wanna say thanks to organizers hosting such a exciting competition. To be honestly, I’d know less about NFL and the rules of American Football Game before this competition started, but now I really enjoy the matches every week;)
And also I appreciate all kaggler sharing info and attending discussions, especially mrkmakr, kenmatsu4, CPMP and charlie_s. Their kernel and insights much affected our models.

Our team is consisted of owruby, David and me. I and David mainly took on the role of feature engineerings and owruby developed NN model. Team members ware fixed just before team merge deadline At that point I and owruby were using Python and David was using R, so main work of the final week was translation from David's features made in R script to Python code. It has been so tough work but produced diversity of features. 

I’ll share brief summary and essence of our approach because it is not so simple and elegant like other high ranker’s solutions.

## Overview
We made a lot of play-level and player-level features. I think our feature’s interesting point is some variations of voronoi features.
Main model is MLP with kinds of attentions and sub-model for ensemble is logistic regression. 

## Validation Schemes
We used two kinds of validations. The first is ordinary Group 5-fold CV with GameId and the second is time-split validation predicting the plays of 2018/12. Each of them don't have consistency to LB much, but when scores of both improved apparently, also LB score nearly always improved. 

## Models
### MLP
MLP model was mainly developed by owruby, so I cannot write down details here. It can be inputted play-level and player-level features separately and then they are concatenated using kinds of attentions. The network is trained by three kinds of losses which is consisted of CPRS loss of Softmax Layer, MAE loss of Linear Layer and CPRS loss of Gaussian Layer with weights of 1.0 : 0.01 : 0.1. The former two losses were from mrkmakr's kernel and The latter one was inspired from kenmatsu4's kernel. Gaussian Layer didn't much improve LB score, but we think it contributed to robustness of training. 

### Logistic Regression
Logistic regression model was trained only by play-level features with year weighted sample (2017 : 2018 : 2019 = 0.1 : 1.0 : 2.0). LR model had not so good score (0.00030 behind to MLP), but weighted ensemble model ( MLP : LR = 1.0 : 0.3 ) improved LB a little (~0.00005). 

## Features
We made various features representing player’s absolute and relative locations, S and A. As others do, we firstly standardized player’s coordinates so that offensive are moving to right side and calculated near future location and speed (0.5sec and 1.0sec after hand-off). And then we created varieties of features like distances, angles between each players and voronoi features with a view of both of offensive and defensive side.

### Voronoi Features
We used some kind features from voronoi calculation. The most efficient feature is x and y lengths of ballcarrier and other's voronoi region at each time point. 
Our voronoi features were originally developed in R by David. A package deldir in R is easily calculate voronoi regions with boundaries, so area of regions and coords of edges are always finite value and it was natural idea to using them as feature. 
However voronoi function in Scipy cannot calculate bounded voronoi regions, so area and coords of voronoi were often calculated as infinite value and I'd strugled to translate calculating process of voronoi features. But thanks to discussion between CPMP and charlie_s in his kernel, I knew I could calculate virtually bounded voronoi regions by making mirrors of the points on four sides and I deployed it. 
The whole voronoi features improved CRPS with about 0.00010.




## Training Augmentation and TTA
We’ve noticed that we can do training augmentation and TTA by flipping coordinates and they improve model performance. But we only used TTA because execution time was limited and using only TTA was almost as effective as doing both.

## Final Submission and Execution Time
We selected 4 seed averaged MLP model and weighted averaging model of 4 seed averaged MLP and 5-fold averaged logistic regression as final submissions. Execution time of the former is ~9000sec with about 4000sec training and the latter is about 10500sec with ~5000sec training. I implemented “try except” everywhere in my code and our kernel submit naive average values in case of error at model prediction or time over.
