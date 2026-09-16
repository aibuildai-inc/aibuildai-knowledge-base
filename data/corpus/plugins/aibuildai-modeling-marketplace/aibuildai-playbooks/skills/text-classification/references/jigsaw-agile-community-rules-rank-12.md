# 12th place solution

Competition: jigsaw-agile-community-rules
Rank: #12
Source: https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/12th-place-solution

Unlabelled data has been used to improve the score by atleast .012. In one of the early probes i found that only less than .5% of the test body values are not from [this](https://www.kaggle.com/datasets/level14taken/jigsaw-2m-reddit-unlabelled/data). This was used later by getting predictions on it from best possible public available [llm ensemble.](https://www.kaggle.com/code/bibanh/lb-0-9-re-finetune-qwen2-5-32b-gptq) Then this set was used to pretrain the deberta-base model with soft labels. This improved the score from .912-> around .924 with 2 seeds just taking around 1hr for submission. 

**Submission Details**
The score was from using Deberta large instead of base to do the same, and keep in ensemble with [triplet with 8x augs](https://www.kaggle.com/code/datafan07/jigsaw-speed-run-10-min-triplet-and-faiss)(.907 public) and [llama 3b with fp16](https://www.kaggle.com/code/wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct/notebook?scriptVersionId=266995301)(.920 public),  public notebooks with slight improvements.

**Sampling From Unlabelled**
 The sampling of data for pretraining was done with claude code interactively, it was just to pick 100 rows from a subreddit see if the rule is violated or not judged by claude iteslf and do this for alll the 6 rules. if the violation rate is less than 1% those subreddits are omitted otherwise it is taken, this way we have 800k points out of 12m possible rows of data for all possible rule-body combinations.
I tried to improve score by variations of data but it didn't work, taking (<1% subreddits) data seemed to hurt the performance of  the model. so they were left out. Taking only bodies of length >50 chars seemed to hurt lb so it was ignored.

**Possible improvements**
The current score was attained in less than 6 hrs. So definitely scope for further improvement.Tried to put another llm to do test time training with the extra 6hrs at hand but could'nt find time to get good hypeparams or to experiment more,  i truly believe this could do well as llm with test time is showing very good progress as seen in the case of llama 3b public notebook. 

**Deberta Training Details** 
followed [this](https://radekosmulski.com/how-to-fine-tune-a-transformer/) as a reference.
Both stages have similar lr scheduler. for stage1 bce with pos_weight was used and training was done for 1 epoch.

**Common Bug noticed in public notebooks** is using bf16 with t4 gpu which makes it run 6x slower.

I thank the organizers and the community on kaggle for their wonderful contributions. 

[Submission](https://github.com/level14taken/jigsaw-comp/blob/main/dinosaur-with-deberta-large.ipynb)
