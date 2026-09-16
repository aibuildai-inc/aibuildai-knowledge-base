# 3rd place solution - GRU, UNET and LGB!

Competition: child-mind-institute-detect-sleep-states
Rank: #3
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459599

# 3rd place solution - GRU, UNET and LGB!

Congratulations to kaggle and the host for this competition, it has been a very interesting and fun competition. Before we start, special thanks to @kyakovlev, it has been exceptional to be able to work with you and learn from you.

## Structure of our approach

**1. Preprocessing**
**2. Training (GRU + UNET + LGB)**
**3. Inference and postprocess**




### 1. Preprocessing
For final submission our GRU + UNET models work just with 7 features. We tried to add more features but for local validation we did not find any that would work.

#### 1.1. **How do we structure the series before introducing them into our model?**
We decided to divide the series into one-day sequences and reduce the granularity from 5 secs to 30 secs. Therefore we had sequences of 2880 length in which there was normally an onset and a wakeup. Here is an standard input for our model.




#### 1.2. **Key points and features in preprocesing**

- Make anglez absolute, this was giving +0.002 on local validation.

- For the only two variables we had (anglez and enmo), we tried to find useful aggregations(diff, mean, median, skew, etc...), but the only thing that seemed to work was the standard deviation (**anglez_abs_std** and **enmo_std**).

- Detecting noise was another of the key points of our solution. We realized that when exactly the same value is repeated in the same series at the same hour, minute and second, this was basically noise. Here our detected noise is the red line.


- To incorporate temporal information into the model, we decided to add 2 frequency encoding variables (one for onsets and one for wakeups) at the hour-minute level.	
- A good augmentation trick was to reverse all the series during training, this allowed us to have more sequences and increased our local validation by 0.01

#### 1.3. **Small ablation study of single model in the middle of the competition**
The public leaderboard has really been making us dizzy throughout the competition, we thought there was a bug in our code. So halfway through the competition we did a study of how each variable was working on the leaderboard and these are the results:




### 2. Training
Our best model was GRU, which alone gave us a local validation of **0.835**

#### 2.1. **GRU and UNET -- training details** (we did the training similar)
- **Augmentation**: just invert series
- **Number of features**: 7 (anglez_std, enmo_std, noise_day_before, noise_day_after, noise, hour_min_onset, hour_min_wakeup)
- **Target**: 2 outputs (one for onsets and other for wakeups)
- **Target transformation**: Add two steps back and one forward. (0,0,0,0,1,0,0,0 -> 0,0,1,1,1,1,0,0)
- **loss** : cross-entropy

#### 2.2. **Leaderboard variability**
It seems that after the competition is over, we can observe that the private is more correlated with our local validation than the public one. In the public there was an instability that was consuming us because we really thought we had a bug somewhere in our code.
Our best private submission is this ensemble: (GRU*0.68 + UNET * 0.2 + LGB*0.12)
CV: ~**0.840** / public LB **0.784** / private LB **0.848**



#### 2.3. **LGB part**
We've been trying to make lgb competitive since we formed a team. We finally managed to start getting him to contribute something to the ensemble the day before the competition ended. So I think that in this part we were a few days away from continuing to refine our LGB model.

Anyway, in our last submission we can see how it was giving us **+0.002** in the private LB.
The maximum we achieved with an LGB single was **0.757** public / **0.82** private.


### 3. Inference and posprocess

For the best submission that we finally chose, we made an ensemble of several models (**8** GRU and **2** UNET)
Running time for inference was ~**1,5** hours.

For postprocess we try several things, but basically what we did was keep the peaks every certain distance optimized for our predictions.



### 4. Conclusions

Surely there are things we are forgetting, so don't hesitate to ask anything and I'll update it in the post.

It has been a type of problem that could be approached in many ways, as we are seeing in the different solutions. We would like to mention that it has been a really challenging competition where it was a constant battle with other teams that pushed us to the limit (everything was very tight). Apart from that, I hope we can help your mission and be useful to the host.

Github code is here: https://github.com/FNoaGut/child-mind-institute-detect-sleep-states-3rd-place-solution
