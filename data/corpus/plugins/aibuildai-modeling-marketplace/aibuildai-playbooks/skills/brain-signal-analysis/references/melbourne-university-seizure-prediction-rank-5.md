# Solution (5th place)

Competition: melbourne-university-seizure-prediction
Rank: #5
Source: https://www.kaggle.com/c/melbourne-university-seizure-prediction/discussion/26117

**We would like to express our gratitude to organizers and all participants.** It was really interesting competition and it would be just great to participate in other similar competitions related to fields of biomedical data analysis.


Here is a description of our solution, which has **AUC 0.81423** in public leaderboard and **0.79363** in private leaderboard.

Software
--------

All data analysis and models were built using Python.  Libraries used: scikit-learn, pandas, xgboost. 

Preprocessing
-------------

The signal from each file was divided on epochs 30 seconds length without any filtration. From each epoch features were extracted. We have tried also 15 and 60 seconds epoch length but the results were worse. 

Feature extraction
------------------

We tried many features in different combinations during this competition, but not all of them were used in final models. **Feature sets** we’ve tried: 
 

 1. [Deep’s kernel][1] for features extraction.
 2. [Tony Reina’s kernel][2] for features extraction.
 3. Correlation between all channels (120 features).
 4. Correlation between spectras of all channels (120 features).
 5. Spectral features version 1: total energy (sum of all elements in range 0-30 Hz), energy in delta (0-3 Hz), theta (3-8 Hz), alpha (8-14 Hz) and beta (14-30 Hz) bands, energy in delta, theta, alpha and beta bands divided by total energy, ratios between energies of all bands.
 6. Spectral features version 2: the same as Spectral features set 1 plus low and high gamma band were used in calculation of total energy, energy in bands and ratios between energies in bands. In addition, mean energy in bands was extracted.
 7. Spectral features version 3: power spectral density was calculated for the whole epoch. Then it was divided on 1 Hz ranges and in each range energy was calculated (30 features).


  [1]: https://www.kaggle.com/deepcnn/melbourne-university-seizure-prediction/feature-extractor-matlab2python-translated
  [2]: https://www.kaggle.com/treina/melbourne-university-seizure-prediction/feature-extractor-matlab2python-translated
