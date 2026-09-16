# 9th place solution: finetune CLIP ViT-H/14

Competition: google-universal-image-embedding
Rank: #9
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359351

First of all, I thank the organizers for this great competition!

My solution is heavily based on @motono0223's work. Thank you very much!
https://www.kaggle.com/code/motono0223/guie-clip-tensorflow-train-example

[* My code is now public *](https://www.kaggle.com/code/akihirok/9th-place-guie-fintune-tf-clip-trainable/notebook)

## Key improvements over the original work

- Input 64D vectors to ArcFace (instead of 256D ones) (~ 0.01 gain)
- [Open CLIP ViT-H/14 on LAION-2B](https://laion.ai/blog/large-openclip/) (~ 0.08 gain)
- Reduce LR linearly (~ 0.01 gain)
- Finetune backbone CLIP model for 1 epoch w/ low LR w/o DA (~ 0.02 gain)
    - 1 epoch = 10 epochs in the @motono0223's work

## Dataset

I indeed thought that finding nice datasets was the most important thing to win the competition.
However, I eventually used the same datasets as the original work.
I couldn't find any other dataset which works well. 😭

Products-10K > Landmark Retrieval 2021 > ImageNet-1K > Deep Fashion = Food Recognition > others (in my best knowledge)

## Note

- I suffered from random OOM errors in submission (with CLIP H/14). I gave up more complex architecture because of this. (why you can do an ensemble or TTA? 🤔)

Thank you for reading!
