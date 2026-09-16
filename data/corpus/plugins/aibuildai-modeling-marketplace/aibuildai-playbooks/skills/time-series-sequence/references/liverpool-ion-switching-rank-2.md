# 2nd place solution

Competition: liverpool-ion-switching
Rank: #2
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153991

UPDATE: All solution code is now available on my GitHub: https://github.com/stdereka/liverpool-ion-switching

**Introduction**

Before I start describing my approach, I would like to give many thanks to the following kagglers:
1. @eunholee for discovering the most effective way of removing drift, explained in this [kernel](https://www.kaggle.com/eunholee/remove-drift-using-a-sine-function).
2. @friedchips for diving really deep into markovian nature of the data. His posts and kernels spurred me to learn more about HMM (it didn't help me in the last weeks of the competition, but I experienced true pleasure learning the subject).
3. @kakoimasataka for attracting my attention to the noise component of the data. His [kernel](https://www.kaggle.com/kakoimasataka/remove-pick-up-electric-noise) allowed me to understand how I should treat the noise in my preprocessing.

There are several things I admitted on the early stage:
1. Preprocessing and data augmentations make more contribution to the final score than model tuning. One of the early breakthroughs of this competition was drift removal.
2. In 0.935+ zone even a very negligible (5 or 6 signs after decimal point) improvement of the score (CV or LB) is important and should be taken into account.

**Preprocessing and data leak**

I described the process of data cleaning and creating new data [here](https://www.kaggle.com/stdereka/2nd-place-solution-preprocessing-tricks). Unfortunately, the private part of the test dataset is corrupted by a leak. It has been already revealed by the other team in this [post](https://www.kaggle.com/c/liverpool-ion-switching/discussion/153824), so I am not going to describe it here. My personal history with this leakage is following. With my way of creating new data, I faced this leak in my local CV and I had to fight with it (I explain how in the modeling section) in order to get reliable CV score. Less than two days before the end of the competition I checked the test data for this kind of leakage. The result shocked me: I could never imagine that I can find it in the private part!

**Modeling**

As a baseline for my final model I used Wavenet-based architecture, which was initially posted in this [kernel](https://www.kaggle.com/siavrez/wavenet-keras). Following ideas worked for me:
1. Early stopping, learning rate scheduler. My model's convergence behavior was slightly different for different folds, so these simple tools helped me to control overfitting.
2. Data augmentation. I used flipping the signal and adding a random shift to the signal. Experiments showed that model predictions are extremely sensitive to a shift, and this augmentation helped the model to became more generalisable.
3. RFC probabilities as features for NN. Initially proposed in this [post](https://www.kaggle.com/c/liverpool-ion-switching/discussion/144645), the effectiveness of this stacking has been confirmed in my experiments.
4. CV strategy. As I trained on different augmented datasets (with different size and target distribution), I had to validate on a subdataset, which would be common for all synthetic datasets. To do so I have chosen the original non-synthetic data. OOF score, obtained on this subset, was used to compare different models, trained on possibly different data. The original train data is not corrupted by the leak mentioned above, so it was safe, and as I got to know after the competition deadline, it was correlated with private LB.
5. One model for all groups of data. In case of NN-based models, it was not necessary to train different models for different groups. Moreover, I observed some sort of synergy between the groups: separate models had lower CV than one general model.

**Final submissions**

1. My first submission is a blend of two models, trained with slightly different scaler and augmentations. No leak is exploited here, this submission was planned before I found the leakage. CV 0.94359, public 0.94664, private LB 0.94529.

2. The second one differs from the first only in 7th batch of the test data. To predict this batch I trained a separate model on reduced signal and then added subtracted channels to its prediction.

**Conclusions**

I have mixed feelings about this competition.

On the one hand, I haven't built a model, which can handle real-world ion switching data. The others published by this moment solutions are also barely applicable to a data different from competition dataset. For example, the drift problem hasn't been solved and all top-scoring models are trained on the clean data. The leak I have found in the last days of competition leaves no confidence in the data. I realized that this kind of leakage can be implicitly hidden in the rest of the data, so complex models can exploit it automatically.

On the other hand, I worked hard for more then a month and a half and enjoyed improvement of my models, it was really funny and cool. Concerning the leak, I think that the task of searching vulnerabilities in the data is valuable in itself. In the long run it can help to build reliable models. I cannot be sure, but probably the knowledge of possible leaks in ion switching data will help electrophysiologists to improve their existing models.

In any case, I am grateful to the organizers and participants of the competition for an unforgettable experience.

**P.S. I am not a native English speaker, so if you find any mistake, please, let me know. If you have any questions and need a clarification on some details, don't hesitate to ask. I'd also appreciate any feedback (positive or negative) you leave in the comments.**
