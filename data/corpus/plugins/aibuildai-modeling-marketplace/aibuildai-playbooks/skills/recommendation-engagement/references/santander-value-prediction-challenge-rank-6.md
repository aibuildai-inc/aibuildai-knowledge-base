# 6th place solution with kernel

Competition: santander-value-prediction-challenge
Rank: #6
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63919

Here is my full single model pipeline starting with just the original 40 features.

https://www.kaggle.com/joeytaj/leak-fe-ml-from-scratch-baseline/notebook?scriptVersionId=5253758

I am very disappointed that the late submission feature is off. Just one more thing that is off about this competition. I really like to use this for wrapping up my learning before I move on and forget all about this problem. 

I had single models (albeit with more feature selection than here) that scored 0.52 private and a good 0.48 public. Since this kernel uses better private leak parameters than I did, it should score pretty close to this. If I didn't introduce any bugs making it.

So most people above me have covered the same solutions I did. Here are the main 2 things I think were important for my score.

1. Removing all the ugly/fake rows before leak predictions.
2. Using the leaks for data augmentation.

Some thoughts on the shake up. Obviously for some there was some major, maybe dissapointing movement. In general I would say there was not unexpected shakup which I am glad for. 

In the end the leak was effective in the private LB.
And just as important the fake data did seem to be actually fake. Not just private test. I was very worried about this until I gave up and went all in on it being fake.

Congratulations and Thanks to all. I would say I am hoping to do better next time. But it I think it takes a bit of luck getting the right problem\solution to get this far. So we will see.
