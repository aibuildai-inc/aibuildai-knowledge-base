# 3rd place solution

Competition: porto-seguro-safe-driver-prediction
Rank: #3
Source: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44608

I have to say, I've been expecting some upward movement on private part of LB, but the final results were really pleasant and surprising!
My solution is not very complex, just an average of 1 lgb and 1 nn, built pretty much on the same feature space. Most important parts:

 1. Feature elimination. I dropped all of calc features and ['ps_ind_14','ps_car_10_cat','ps_car_14','ps_ind_10_bin','ps_ind_11_bin',
             'ps_ind_12_bin','ps_ind_13_bin','ps_car_11','ps_car_12']. I was excluding them one by one in greedy fashion and checking lgb cross validation score.
 2. Hot encoding categorical variables. It helped to reduce noise while getting the splits for most useful categories.
 3. For NN model it was also necessary to hot encode numeric features with small number of unique values - ['ps_car_15','ps_ind_01','ps_ind_03','ps_ind_15','ps_reg_01','ps_reg_02'] (without dropping the original ones)
 4. Regularized models. lgb_par = {'feature_fraction': 0.9,  'min_data_in_leaf': 2**4, 'lambda_l1':10,
       'bagging_fraction': 0.5, 'learning_rate': 0.01,  'num_leaves': 2**4}

Another thing that I unfortunately haven't explored well is anomaly detection on train+test datasets. Just like less frequent categories (or combinations) of categories are more likely to have label 1, we could find 'strange' samples via unsupervised methods. For example, if we train a basic autoencoder, AUC score of sample-wise reconstruction error would be ~0.60, which is pretty high. I believe more thorough analysis could make this approach really useful. 

Generally, it's hard to tell what else did not really work or could have worked: almost everything that you will try to do in a competition like this will result in no significant change, no matter whether you did it right or wrong:)

`sellout`

If you want to learn useful ML techniques and competition specific tricks, check out this course https://www.coursera.org/learn/competitive-data-science/home/welcome from experienced kagglers like KazAnova and me.

`/sellout`
