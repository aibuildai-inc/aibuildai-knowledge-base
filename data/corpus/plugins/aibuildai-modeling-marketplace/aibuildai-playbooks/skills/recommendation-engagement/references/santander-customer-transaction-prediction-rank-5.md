# 5TH Place Solution

Competition: santander-customer-transaction-prediction
Rank: #5
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88897#latest-517607

Congrats to the winners and the hosts – this was a fun competition. I would also like to thank my teammates for doing a great job trying to decipher the data. 

I would have preferred 6th rather than 5th - now we have to reproduce the results! 

In the beginning of this competition I was relieved because I could not see any leakage – little did I know! I could not have been more wrong! I would guess that the top solutions don't add much value for this problem as everybody was exploiting the leakage - maybe in the future the organisers need to be more careful with the data. 

On the other hand, if there was no leakage, it wouldn't be  interesting!

To the matter at hand.

#### Leakages

1. [Half of the test data is fake](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/85125#latest-511175) 

2. Count of unique values of all features, of train + real test data together. If you also include the fake data in the calculations, the uplift will **NOT SHOW** on LB. In other words you need to create 200 extra features representing the count of unique values for all numerical features. E.g they have some categorical properties. This can get a lightgbm model to the `0.914x` level.

3. Augmentation. Since features are uncorrelated (also explained by Braden [here ](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/83882#latest-499474) you could generate multiple artificial samples. The augmentation we used was similar to [this ](https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment)   . What made all the difference is combining all previous 1,2. E.g when you do the augmentation and you take random values from one column, you need to make certain that you also sample the unique counts for these values for that column (as computed from the combined train + real test). Trying different augmentation- ratios ( we found increasing the positives 20 times and the negatives 2 times to be around the best for maximum lb performance), you could get lightgbm to `0.920x`. 

#### Magic

1. `NN`s perfomed better than `lightgbm`s, but needed an input shape of  `Input(shape=(200,2))`. The first one represents the features and the second the unique count associated with these features. They could easily reach `0.924` in public LB with this transformation + augmentation. This nn's architecture and augmentation can be seen [here](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88929#latest-514001): 
2. For `lightgbm` it helped using `minmaxscaler` of (-4,4) on the original features and then doing `Xn**countn` where the count is clipped between 1 and 3
3. Again for `lightgbm` it helped using `minmaxscaler` of (-4,4) on the original feature and then doing `countn** Xn` where the count is clipped between 1 and 3
4. Alaso `xn**-3` also helped a bit
5. Also very important to get almost another +0.0005 was pseudo-labelling

We were trying to build a different model for each column, but it did not work out very well.

Stacking added around +0.001. NN was significantly better at stacking than any other method.
