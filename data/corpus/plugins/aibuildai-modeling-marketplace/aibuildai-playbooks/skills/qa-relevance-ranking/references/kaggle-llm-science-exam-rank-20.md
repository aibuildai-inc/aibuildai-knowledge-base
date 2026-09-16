# 20th Place (Top 1%, Solo Silver) Solution

Competition: kaggle-llm-science-exam
Rank: #20
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/448114

First and foremost, I would like to express our gratitude to the hosts and the Kaggle team for arranging this wonderful competition. Also, a huge shout-out to all the participants for fighting through this intense competition. It was an extremely educational and exciting competition.

As my solution is quite similar to the top teams and I'm a little busy these days, I didn't work on the writeup until I was mentioned by @kaggleqrdl in this [discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447396#2486398). It's time to spend some time to finish this writeup. Thanks @kaggleqrdl for memorizing me in this competition:)

My final solution is quite simple, 4 RAG + 1 deberta-v3-large ensembling:

[llm-solution]

can improve to 0.917 if change mean ensembling to max + mean ensembling. Refer to my submission [code](https://www.kaggle.com/code/wuwenmin/llm-science-ensemble-exp-max/notebook)

**Things that didn't work for me**
* Pre-finetune on RACE
near the end of the competition, I tried to pre-finetune the model on RACE and then finetune on 60k with much smaller LR, but it didn't improve the CV & LB

**Special thanks to** 
* @cdeotte, @mbanaei, and @yalickj  for providing us the valuable datasets.
* @ksmcg90 for providing a faster data loading method using pyarrow (refer to his [notebook](https://www.kaggle.com/code/ksmcg90/faster-context-extraction-pyarrow-fixed))
* @itsuki9180 for providing a solution to training using deepspeed (refer to his [notebook](https://www.kaggle.com/code/itsuki9180/using-deepspeed-with-hf-trainer)). I trained the deberta-v3-large on 8xRTX4090, the speedup was amazing:)
* @datafan07  for providing a way to finetune LLMs for sequence classification using RewardTrainer (refer to this [notebook](https://www.kaggle.com/code/datafan07/single-model-rewardtrainer-lora-llm/notebook))
* and all the teams for sharing their valuable solution for this task. I need spend one more weekend to learn all the techniques:)
