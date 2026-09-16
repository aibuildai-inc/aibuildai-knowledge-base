# 4th place

Competition: birdclef-2022
Rank: #4
Source: https://www.kaggle.com/c/birdclef-2022/discussion/326987

Thanks to the organizers for another sound competition. My journey to kaggle started with your bird competition and I am becoming a Grandmaster in your bird competition as well. It's sad that a model trained on non-public data won, because of which, it just couldn't be beaten. 

I didn't have much time to participate, so my solution is overfitting last year's model [2020](https://www.kaggle.com/competitions/birdsong-recognition/discussion/183269)
 [2021](https://www.kaggle.com/competitions/birdclef-2021/discussion/243351)

Differences:
1. First I train on all birds, then I finish training on 21
2. I select a different lb threshold for each species.

Otherwise, my decision repeats my public decision of previous years
1 fold = 0.78 private lb
