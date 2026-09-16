# [33rd place] Key points of my solution

Competition: equity-post-HCT-survival-predictions
Rank: #33
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566572

### 1. Target Construction
While many participants used Kaplan-Meier fitted targets, they overlooked its fundamental meaning. The Kaplan-Meier curve represents survival probability at given time points, whereas we need to measure patients' "risk index". Direct application ignores critical differentiation between efs=1 and efs=0 samples when efs_time is identical.

Key insights:

For efs=1 samples: Survival probability at efs_time directly reflects risk magnitude
For efs=0 (censored) samples: We need to estimate their expected event time
Solution framework:
Define a critical time t where:
P(event at t | survival until efs_time) = 0.5

Through conditional probability derivation:

P(event at t | survival until efs_time) = P(event at t) / KM(efs_time) = 0.5
Where KM(efs_time) is the Kaplan-Meier survival probability at efs_time.

Thus for censored samples (efs=0):
Final target = 0.5 × KM(efs_time)
This enables unified risk evaluation across both sample types.

### 2. Sample Weighting
While establishing risk indices for censored samples, we must acknowledge their inherent uncertainty compared to efs=1 samples - particularly those with low efs_time values.

Experimentation:
Attempted efs_time-based dynamic weighting (unsatisfactory results)
Final approach: Global weight of 0.6 for all efs=0 samples


While pursuing technical excellence in competitions, let's remember the real-world impact of our work. May all allogeneic HCT patients receive optimal care and embark on journeys filled with hope and fulfillment. The true victory lies in translating these data patterns into better clinical outcomes.
