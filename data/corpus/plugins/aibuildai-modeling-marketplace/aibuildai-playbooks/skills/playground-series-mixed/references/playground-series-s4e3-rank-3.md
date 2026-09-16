# 3rd Place Solution - Mediocres et Impera

Competition: playground-series-s4e3
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s4e3/discussion/488127

### Context:

Steel Plate Defect Prediction Competition (Playground Series - Season 4, Episode 3)

- Business Context: [https://www.kaggle.com/competitions/playground-series-s4e3](https://www.kaggle.com/competitions/playground-series-s4e3/data)
- Data Description: [https://www.kaggle.com/competitions/playground-series-s4e3/data](https://www.kaggle.com/competitions/playground-series-s4e3/data) 

### Overview of the approach:

Due to data and targets nature of this competition I considered competition as a multi-label problem. I've tried also multi-class approach but CV and public LB scores were a bit lower comparing to mutli-label. 
Final submission is a combination of the several models - PyBoost, AutoGluon and some public bests submissions. 
My simple secret souce in this competition is "Mediocres et Impera" (average and rule) strategy: which here means - take something you trust the most and average the results. 

### Overview of the submissions:

- Selected submission: 0.88936 (Private, **3rd** place), 0.89674 (Public, 8th place) - Mix
- Selected submission: 0.88763 (Private), 0.89476 (Private) - PyBoost + OpenFE (features)
- Unselected BEST: 0.88944 (Private, **2nd** place), 0.89670 (Public, 8th place) - Mix

### What works?

- Multi-Label approach
- PyBoost framework for really quick multi-label approach (thanks to @btbpanda)
- AutoGluon AutoML (thanks to @innixma)
- Ideas, notebooks and submissions published by @thomasmeiner, @arunklenin, @ravi20076, @lucamassaron, @liudacheldieva, @ivanblch, @oscarm524 
- OpenFE framework for features generation
- Merging, Stacking, Ensembling, Averaging
- 🦄 Unicorn's pollen

### What doesn't work this time?
- Standalone models 
- Original Data

### Sources:
[https://github.com/sb-ai-lab/Py-Boost](https://github.com/sb-ai-lab/Py-Boost)
[https://github.com/autogluon/autogluon](https://github.com/autogluon/autogluon)
[https://github.com/IIIS-Li-Group/OpenFE](https://github.com/IIIS-Li-Group/OpenFE)
[https://www.kaggle.com/competitions/playground-series-s4e3/discussion/481015](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/481015)
[https://www.kaggle.com/competitions/playground-series-s4e3/discussion/481098](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/481098)
[https://www.kaggle.com/code/arunklenin/ps4e3-steel-plate-fault-prediction-multilabel](https://www.kaggle.com/code/arunklenin/ps4e3-steel-plate-fault-prediction-multilabel)
[https://www.kaggle.com/code/ravi20076/playgrounds4e03-eda-binaryclassifier](https://www.kaggle.com/code/ravi20076/playgrounds4e03-eda-binaryclassifier)
Thanks to the authors 🙏


####BONUS Leaderboard Visualisation
[https://www.kaggle.com/code/samvelkoch/s4e3-shake-up-plot](https://www.kaggle.com/code/samvelkoch/s4e3-shake-up-plot)
