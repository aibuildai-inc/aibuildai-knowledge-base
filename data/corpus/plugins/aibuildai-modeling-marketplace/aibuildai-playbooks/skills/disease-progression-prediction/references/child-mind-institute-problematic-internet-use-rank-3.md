# Child Mind Institute PIU 3rd Place Solution

Competition: child-mind-institute-problematic-internet-use
Rank: #3
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552677

First of all, I’d like to thank the organizers for hosting this competition and everyone for making it such a thrilling experience. Despite the challenges posed by unpredictability, it provided a valuable opportunity to learn how to build robust solutions for small, noisy datasets.
My approach was straightforward, and I’m excited to share it with you.

**Cross-Validation**
One of my key focuses, like many others, was to establish a stable and reliable CV framework. I avoided using any fixed random seed throughout the process. It took me 100 repetitions of 5-fold stratified KFold to achieve stable results, and I used 20 repetitions during Optuna hyperparameter tuning.
To optimize the final QWK threshold, I used the OOF predictions from all these repetitions.

**Model**
I stuck to LightGBM for the entire competition. I did start working on a CatBoost solution at one point but lost the energy to take it further or combine the two.

**Feature Engineering**
*Actigraphy Data*:
-Calculated the standard deviation for X, Y, Z, and AngleZ, and the mean for Elmo.
-Derived features representing the five longest streaks of inactivity and activity using Elmo.
-Binned the "light" column into categories ranging from twilight to direct sunlight and took the value counts for each category.
*Instrument Data*:
I started with features from the public notebook, checking each one to see if it actually contributed to the model. After that, I added a handful of custom features based on my own experimentation.

**Data Augmentation**
*NaN Augmentation*:
Initially, I imputed NaNs randomly in columns that already had missing values. Eventually, I just imputed NaN on all columns with NaNs for 20% of the data and combined this augmented data with the original dataset.
*Gaussian Noise and Imputation*:
I applied simple imputation and added Gaussian noise to 20% of the data. This augmented data was then merged with the original dataset.

**Post-Processing**
I used the 'PCIAT-PCIAT_Total' column for training. To finalize predictions, I applied the optimized threshold to calculate the sii for each of the 100*5 models and took the mode to generate the final predictions.

**Results**
Initially, my CV-LB correlation started to break down after achieving a leaderboard score of 0.46. At that point, I decided to focus entirely on CV and improve it further. I’m pleased with the results of this phase, which led to consistent private leaderboard scores. Below are the highlights from my last 5 submissions during this phase:

LB Score     PB Score	    Repeats
  0.445	       0.477	         1
  0.461	       0.482	      100
  0.461	       0.479	      100
  0.466	       0.478	      100 (best submission selected)
  0.458	       0.480	      100
All of these submissions had nearly identical CV performance:
Validation QWK: 0.454 - 0.456
Optimized QWK: ~0.470

After this phase, I switched strategies by fixing the random seed and focusing on achieving higher LB scores with minimal changes. While this led to a slight improvement in CV—validation QWK around 0.460 and optimized QWK around 0.471—the LB scores remained the same, and PB scores worsened, averaging around 0.470 on the private leaderboard.

That’s all, thanks for reading!

https://www.kaggle.com/code/jobayerhossain/child-mind-piu-3rd-place-solution
