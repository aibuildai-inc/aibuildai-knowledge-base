# 17th Place Solution| AutoML + Unicorn's pollen + Lack of sleep

Competition: playground-series-s4e1
Rank: #17
Source: https://www.kaggle.com/c/playground-series-s4e1/discussion/472636

## Context

S4E1 Playground "Binary Classification with a Bank Churn Dataset". 

- Business context: [https://www.kaggle.com/competitions/playground-series-s4e1/overview](https://www.kaggle.com/competitions/playground-series-s4e1/overview)
- Data context: [https://www.kaggle.com/competitions/playground-series-s4e1/data](https://www.kaggle.com/competitions/playground-series-s4e1/data)

## Overview of the approach
Our final submission was a combination of AutoGluon 3-level stack we called "Frankenstein II" and set of  averages from our previous models and some public notebooks. 

Final submission was trained on the reduced set of features we got from OpenFE. Features were eliminated by BorutaSHAP and RFECV. Final model used 103 features. 

## Detail of the Submissions
We selected 2 submissions: 
- WeightedEnsemble_L3 0.89372 Public | 0.89637 Private | 0.898947 CV
- Winning solution  0.90106 Private | 0.89687 Public. We got it from averaging 0.89673 and 0.89565 in last hours of the competition. 

## Frankenstein II schema
.png?generation=1706805159458796&alt=media)

## What worked for us?
- Feature generation - 470 and Feature Elimination - 103
- Data-Centric Approach (CleanLab)
- Relabeling 
- AutoGluon 1.0.1 (thanks to @innixma)
- BorutaSHAP framework and Skleran - RFECV
- Ideas published by @paddykb, @thomasmeiner and respected community
- Merging, Stacking, Ensembling, Averaging
- Tons of experiments. Mainly for educative purposes
- 🔥 **Kaggle Alchemists Secret Society named after Akka från Kebnekajse**
- 🦄 Unicorn's pollen 

## What doesn't work for us this time? 
- PCA / ICA
- Standalone Boosting models
- TabPFN
- Surnames features
- Original dataset


## Sources
- [https://www.kaggle.com/competitions/playground-series-s4e1/discussion/470363](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/470363)
- [https://www.kaggle.com/competitions/playground-series-s4e1/discussion/471164](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/471164)
- [https://www.kaggle.com/competitions/playground-series-s4e1/discussion/469859](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/469859)
- [https://www.kaggle.com/competitions/playground-series-s4e1/discussion/465192](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/465192)
- [https://www.kaggle.com/competitions/playground-series-s4e1/discussion/470610](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/470610)
- [https://www.kaggle.com/code/arunklenin/ps4e1-advanced-feature-engineering-ensemble](https://www.kaggle.com/code/arunklenin/ps4e1-advanced-feature-engineering-ensemble)
- [https://www.kaggle.com/code/thomasmeiner/ps4e1-eda-feature-engineering-modelling](https://www.kaggle.com/code/thomasmeiner/ps4e1-eda-feature-engineering-modelling)
