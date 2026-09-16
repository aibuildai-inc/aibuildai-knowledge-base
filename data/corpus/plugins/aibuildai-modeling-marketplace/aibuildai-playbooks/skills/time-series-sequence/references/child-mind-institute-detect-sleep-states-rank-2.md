# 2nd Place Solution

Competition: child-mind-institute-detect-sleep-states
Rank: #2
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459627

Thanks to organizers for this interesting challenge and congrats everyone who enjoyed it! It was a fun competition though the data is very simple 1D data. I look forward to seeing the various solutions.

# Overview of my solution
My pipeline consists of:
- **First Stage**
  - Preprocess and Feature engineering
  - Event Detection and Sleep/Awake Classification by 1DCNN(U-NET)
  - PostProcess of 1D CNN (such as Peak Detection)
- **Second Stage**
  - Use ML instead of the Post-processing that is difficult to adjust
  - Rescoring by LGBM that can consider the limitation of 2evets/day
- **Third Stage**
  - Predict as many events as possible just by shifting the step of the base prediction above.
  - Scoring the generated predictions by LGBM
- **Ensemble**
  - combination of averaging and WBF ensemble

I think the 2nd stage is the most unique. 
This model predict the daily accuracy curve of candidates received from 1st stage. 



Details are below.

# 1st Stage: Event Detection and Sleep/Awake Classification



# 2nd Stage: Rescoring the confidence considering the limitation of 2 events/day
## Concept
I applied 2nd stage for the following 2 reasons.
### 1. Less than 2 events/day
The first stage NN cannot consider the very important limitation of this competition that the event occurs at most twice in a day. Which do you think is important?
> A) The third candidate in a day with a confidence score of 0.20.
> B) The first candidate in a day with a confidence score of 0.19.

In many cases, the latter is more important than the former though the score is lower.

### Long-term features
NN is really strong. However, it’s not good at handling meta features and long-term cyclic features. For example,
“He can be sleeping longer than the usual. This might not actual wakeup.”
“This onset is relatively unclear compared to the other days.”

## How to train 2nd stage model

# 3rd Stage: Add as many events as possible


# Ensemble
  - prepare 2 CNN models
  - Averaging 10(5fold x 2seed) predictions at 1st stage for each model
  - Run 2nd stage for each model
  - WBF-like Ensemble of 2 models
  - Run 3rd stage


# Ablation Study
not yet
- CV Scores
  - 1st stage: 0.826 (-0.019)
  - add 2nd stage: 0.832 (-0.012)
  - add third stage: 0.842 (-0.002)
  - add model ensemble(2models): 0.844 (baseline. final submit)

### Inference
https://www.kaggle.com/code/kmat2019/cmi-sleep-2ndplace

### Training
https://www.kaggle.com/code/kmat2019/cmisleep-training-sample-2ndplace-kmat
