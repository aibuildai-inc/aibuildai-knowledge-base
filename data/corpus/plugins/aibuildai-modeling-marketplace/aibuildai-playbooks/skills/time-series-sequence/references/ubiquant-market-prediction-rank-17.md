# 17th Place Solution, 1000+ on public lb

Competition: ubiquant-market-prediction
Rank: #17
Source: https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338239

[Link to Model](https://www.kaggle.com/code/petersk20/17th-place-solution/notebook)


**Model Design:**
- Used all the features except for ['f_170','f_272','f_182','f_124','f_200','f_175','f_102','f_153','f_108','f_8','f_145', 'f_225', 'f_241', 'f_63', 'f_229', 'f_246', 'f_41', 'f_66', 'f_142', 'f_150', 'f_99', 'f_74', 'f_62', 'f_271'] which were removed using permutation importance.

- Chinese stocks are infamous for having suspensions in trading.  So, I engineered a feature called missing which checked if the stock was present one time step before.  My model using this feature scored 0.117721 vs 0.115486 without it (my second submission was very similar but without this variable).  This is a small difference but would have dropped me to 84th place.  

- Only used stocks after the 850th time interval and weighted newer time steps more.

- Scaled the target using the mean and std within each time step independently of other time steps. 

- Applied batch norm to all the features except for missing and put the features into a network with two hidden layers (1000, 512).

- I trained for 21 epochs.  I went through the training set twice per each epoch.  The first run used l2 as the loss and a batch size of 128 where as the second run used l1 loss and variance with a high batch rate of 1000.  The two run throughs also had different learning rates.  The first one had a decaying lr and the second one had a small lr that didn't decay (.00002 vs .0000006).  I got the predictions on the 11th, 16th, and 21st epoch and averaged the predictions to get the final.

**Takeaways:**
- I wish I had used a lightgbm model to blend with my NN.  I had seen a good improvement in the public lb with this strategy but didn't have the time to tinker with a lightgbm model to get it to not cause memory issues.

- I also learned from the G-Research competition not to run too many experiments to find optimal values for the hyperparameters and other design decisions of the model due to the low signal to noise ratio present in financial markets.  I think this helped me not to overthink the design of the model and do a bunch of work for little gain. 

- I'm also glad I trusted my model because my score with this notebook on the public lb was 0.1487 while most people were at least getting over 0.150.
