# 22th place solution

Competition: liverpool-ion-switching
Rank: #22
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/154705

## My first silver medal! 
## And I became an Kaggle Expert!
Thank you very much!!!!!

I'm a little late in publishing my solution.
I'm sorry.

I've tried a few ideas, but I'd like to highlight the one that worked the best for my score in particular.  
The other ideas I have are ones that others will surely be trying.

I was very affected by this notebook (https://www.kaggle.com/jt120lz/open-channel-clear-plot).
Thank you, liuze!!!!


### train data
The following graph shows the signal replaced by the average value per openchannels. For example, the red line in Batch5 shows the average value of openchannels = 5 for signal in Batch5.

.png?generation=1590758184980770&amp;alt=media)

As you can see, Batch4 and Batch9 clearly behave differently than the other batches.  
I've corrected this discrepancy.
This is the only thing to do with the train data.
.png?generation=1590758231124352&amp;alt=media)



### test data 

The group is defined as follows.


The problem is test data. test data doesn't have open_channels information, so the above method can't be used.

However, what needs to be done is immediately apparent when you look at the kde plot.
Compare the test data(group4) with the modified train data(group4).




It's clear that we have work to do. 
Just fix the above misalignment.  


That's nice.


## code
All you have to do is process the following!
Thanks!!
```python
# --- train ---
off_set_4 = 0.952472 - (-1.766044)
off_set_9 = 0.952472 - (-1.770441)
# batch4
idxs = df_tr['batch'] == 4
df_tr['signal'][idxs] = df_tr['signal'].values + off_set_4
# batch9
idxs = df_tr['batch'] == 9
df_tr['signal'][idxs] = df_tr['signal'].values + off_set_9

# --- test ---
off_set_test = 2.750
df_te['signal'] = df_te['signal'].values
idxs = df_te['group'] == 4
df_te['signal'][idxs] = df_te['signal'][idxs].values + off_set_test
```
