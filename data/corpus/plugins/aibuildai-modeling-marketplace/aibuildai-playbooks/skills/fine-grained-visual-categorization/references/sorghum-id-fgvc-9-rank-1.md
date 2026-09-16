# 1st Place Solution

Competition: sorghum-id-fgvc-9
Rank: #1
Source: https://www.kaggle.com/c/sorghum-id-fgvc-9/discussion/329049

Thanks to Kaggle and the hosting team for an interesting competition.
# summary
My method is very similar to that of the third place([sorghum-id-fgvc-9/discussion/328593](https://www.kaggle.com/competitions/sorghum-id-fgvc-9/discussion/328593)).

The main differences are as follows:
1. I used convnext base network as backbone.
2. I trained 5-folds models and ensembled  them.
3. I combined this contest data with fgvc8data to generate a new dataset with 223 categories instead of 100.
# results
|  step|  method|Public Score|Private Score|
| ---| --- | --- | --- |
| 1| convb+100categories | 0.953 |0.952|
| 2| convb+223categories | 0.955 |0.954|
| 3| ensemble(step1+step2) | 0.959 |0.957|
| 4|step3+pseudo label| 0.963 |0.962|
| 5|step4+5folds| 0.965 |0.965|
