# 5th place solution

Competition: child-mind-institute-detect-sleep-states
Rank: #5
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459766

Thanks Kaggle and competition host for this interesting competition that can be approached by various of methods, it is an enjoyable journey to explore this dataset.
And also great thanks to people who sharing insights and ideas during the process of this competition, I learnt a lot from you all.

**overview**
I approach this problem in four stages: 
1. candidate step generation by heuristic rules 
2. step correct model to improve step quality 
3. score model generate score for submission 
4. post-processing 


**candidate generation**
basic idea of heuristic rules, you can check code for more details and tricks.
- region with |diff(anglez)|<5: start -> onset, end -> wakeup. 
Onset/wakeup is not likely to happen in the middle of an inactive region. Region with suspicious fake data (rules to detect such region is described later) is removed.
- fake region extended by inactive region: start->wakeup, end->onset. 
One explanation is that behavior of removing the watch can only happen when one is awake.
- current candidates shift by +/-720 when no other candidates in range of 720*2.

The three rules generate 254653 candidate steps in total for all series with best possible score 0.9006.


**step correct model**
- Lightgbm L1 regression with (nearest_target_step-step) as target
- since we just interested in the quality of steps that matched with target, weight of data points with |target|>=360 is set to be zero
- data points is weighted by threshold_class_width^(-6) to focusing on ones nearing the target, as I don’t think accurately predict the gap is possible when true target is far away.（threshold_class_width eg: 120<y=127<150, then resp class_width is (150-120)=30）There are also many other choice of weights that can reach similar performance based on my experiments.
- As minute%15 and second are the most important features I think this stage is mainly correct for the bias in label generation process rather than actually improve the step quality. And the most gains of this stage actually come from shift step by ~-11, which is not reflected in feature importance list.
- To validate next stage without leak CV is done in inner fold with 5fold group-k-fold by series_id, so 2* 5* 5 models in total.

Best possible score after this stage is 0.932.


**score model**
- Lightgbm with cross-entropy objective
- target=max(0,1-|nearest_target_step-step|/360) for data points nearest to each target else target is set to zero.
- negative data points (target=0) is down weight by *0.7/0.4 for those with |nearest_target_step-step| >/<360.
- target is created on corrected step given by out of fold prediction in last stage.
- I create different models for onset and wakeup candidates and use 5fold group-k-fold by series_id to do validation, so 2*5 models in total.

I compute competition metric after post processing, so no score is reported here, but It is easy to add one.


**post-processing**
this consists of three parts:
- if step%12=0 then step+/- 1 to match more target steps.
- for two candidates with gap < 720 only keep them when:               
`min(|gap|/720,1)*(exp_score1*exp_score2/(exp_score1**2+exp_score2**2))**0.5>0.083`
here 
`exp_score=np.exp(np.arctanh(2*score-1))`
else remove the one with smaller score.
This avoid too dense candidates and only works when model preference to candidates is clearly biased.
- If score sum in day for some series is larger than 1 then all scores in that day will be divide by score sum.

After post-processing CV score 0.825, public LB score 0.783, private LB score 0.844.


**more details**
- **fake region**
  1. anglez 5min mean (6 decimal accuracy) appear more than one time in the same series
  2. anglez 5min std>0.5 to avoid mark some inactive region as fake
  3. mark regions surrounded by long fake region as fake, as described in GGIR website.
- **data cleaning**
  1. remove some series with many wrong labels by manual inspection
  2. data points covered by long time unlabeled tail region is set to weight zero
  3. data points related to target with unexpected anglez distribution is set to weight zero. (onset followed by active region, wakeup followed by inactive region )
  
  Those cases are only removed from training process, I still include them when compute CV score.
- **features**
  
  2040 in total, I only describe the most important ones, the remaining ones can be duplicated, unimportant or useless. You can check code for more details.
  1. window based features: **source_column X stats X window_size X window_operations**, not full combination.
  **source_column:**  *anglez_abs_diff, enmo_abs_diff, anglez_abs_diff_quantile, smoothed_anglez_abs_diff, 
  anglez_abs_diff_in_given_range, fake_mark, sleep_mark by GGIR heuristic rules, …*
  **stats**:  *mean/50 quantile/95 quantile/max/min*
  **window_size**:  *1min,3min,5min,10min,15min,30min,1h,2h,4h,8h,12h*
  **window_operations**:  *left_side_window(lw), right_side_window(rw), (lw-rw), [(lw-rw)/(lw+rw)], concat(lw,rw), [(lw-rw)/ concat(lw,rw))]*
  2. time features
weekday, hour, second, minute, second_in_day, step/max_step
minute%15 (as discussed in https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/444374)
  3. feature stats aggregated in day
mainly fake_mark and sleep_mark
  4. step gap to mean event time in series given by heuristic methods
  5. stats values at the step time
  6. step gap to nearby candidates and feature value gap with some features that ranked high based on feature importance, manually picked.


**Reproduce**
re-run following notebooks in order
https://www.kaggle.com/code/w5833946/cmi-lgb-v9-train-reproduce
https://www.kaggle.com/code/w5833946/cmi-lgb-v9-predict-reproduce
