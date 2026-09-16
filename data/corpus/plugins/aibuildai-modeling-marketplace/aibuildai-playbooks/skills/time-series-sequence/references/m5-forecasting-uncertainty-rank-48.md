# 48th place solution - seq2seq

Competition: m5-forecasting-uncertainty
Rank: #48
Source: https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163389

First of all, congratulations to all the winners and thanks to everyone for sharing their insights during and after the competition. This was my first forecasting competition/project so I learned a lot of new things.

My best solution, which got me the 48th rank, was a simple ensemble of two architectures - a Dilated LSTM Sequence2Sequence model and a Sequence2Sequence LSTM model with attention (applied from all encoder outputs to hidden state of each step in the decoder).

I have shared my code for both the accuracy and uncertainty competitions on github (https://github.com/arshjot/Kaggle-M5-Forecasting) and will share some more details below.

### **Features**
- Calendar features (42):
 * Cyclic encoding for month, day and weekday (as explained here - https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning) - 6
 * Relative year (2016 - year) - 1 
 * Embedding of size 16 for event_name. Both the event fields were passed through the same embedding layer and later concatenated - 32
 * Snap fields with values as 0 or 1 - 3

- Series IDs (63):
Embeddings for all id fields  - item\_id (50), dept\_id (4), cat\_id (2), store\_id (5), and state\_id (2). In case of aggregated series (Levels 1-11), the grouped fields are kept as separate categories ('All'). For example - Level 2 series of state CA is categorized as:

| item_id | dept_id | cat_id | store_id | state_id |
| -- | -- | -- | -- | -- |
| All | All | All | All | CA  

- Previous day sales
- Sell Price (for levels 1-11, mean sell price of their constituent series was used)
 
### **Normalization**
Only price and sales features were normalized. This was done by dividing the sales and price of each series by their means in the input window

### **Training**
A sliding window of length 28\*13 days on last 2 years data (with a step size of 28 days) was used for training. This meant for each window, 28\*13 timesteps were given as input to the encoder and the succeeding 28 days served as the prediction horizon for the decoder.

SPL loss was used for training with RMSProp optimizer, reducing learning rate, and early stopping.

I also ended up using some lag/rolling features and addition of noise during training though these modifications did not result in a significant improvement in the validation score 

### **Validation**
A simple 3-fold conventional time-series validation split (last 3 periods of 28 days) was used, with the predictions from each fold averaged for the final submission.

---

Interestingly, my models performed better in the uncertainty stream than the accuracy one both in terms of validation and leaderboard scores.

I also wanted to try representation learning and graph learning algos (to account for hierarchy) but could not devote enough time to make them work.

I hope my post/code turns out to be useful for someone. Please let me know if there are any questions :)
