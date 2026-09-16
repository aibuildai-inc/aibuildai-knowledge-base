# 2nd place solution

Competition: google-smartphone-decimeter-challenge
Rank: #2
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/261876

Thank you hosts for such an exciting competition!
My final solution can be described as follows:



and I was inspired by Smart PPP algorithm presented [here](https://link.springer.com/article/10.1007/s10291-021-01106-1)

1. XGBoost stacked ensemble improved baseline
I've spend about 3 weeks at this point and started with idea of correcting Baseline PrMs using other device data. This was jumpy but after all it works. Each stacked ensemble model took about 5 hour to train with GPU, and it was hard to validate. In my opinion this gives about 10-40 cm improved baseline. No further smoothing Pseudoranges was applied. Technical trick: amount of correction of PrM was used as weights in WLS algorithm

2. Computing velocity
I've spend about week on translating WlsPvt.m from gps-measurement google tool in matlab algorithm to python using PseudorangeRateMetersPerSecond and it was 0.08Mps mean average accurate. Technical trick: When algorithm failed I used baseline speedMps described in my notebook position shift. This results improved a lot kalman filtering and mean by stop technique.

3. ADR relative position
I've spend about week on computing relative position using ADR and I didn't succeeded, thus I've used derived files public version Android GPS tools. It took a lot of time to process raw gnss files in this software, and failed for two phones from test set. Technical trick: some of automatically downloaded ephemeris files were corrupted, thus it works to download them manually and turn off internet for the processing time.

4. Remove outliers
I took idea from @dehokanta but it was upgraded. Technical trick: If there were two or more consecutive outliers remove and interpolate it

5. Mixing signals from other devices:
My very first submissions was starting here. It turns out then median averaging is better then mean, but I've ended with idea found [here](https://stackoverflow.com/questions/55813719/multi-sensors-fusion-using-kalman-filter): which is described as follows:

Technical trick: variance among all epochs is pretty similar so I windowed signal (10-30 epochs adjusted by speed Category and region) 

6. Mean by stop
Just using velocity Mps with threshold adjusted about 0.5-1. At this point (Xgboost + postprocessing: remove_outliers + mixing + kalman + stop) I've reached 3.945 public and 2.379 private

7. Snap to ADR shape
I used mean adr position - mean position during valid cycle as shift and shifted relative positions. Technical trick: This shift was weighted by adr uncertainity meters (public 3.345 private 1.903)


8. Snap to ground truth
At this point i've started to snapping this solution to ground truth (threshold 2m for high speed and threshold way more then 2m for SJC) Technical trick: I've selected SJC regions by mean speed < 6 Mps total. (best public 3.159 private 1.801)

0. Technical trick: joblib + optuna were so useful for me among almost all steps

After all, using ~~GPS~~ GNSS will never be the same for me thus I've started as a completely newbie - thank you!
