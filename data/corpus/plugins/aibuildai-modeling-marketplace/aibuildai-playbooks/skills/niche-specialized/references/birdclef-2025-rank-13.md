# 13rd solution for BirdCLEF+ 2025

Competition: birdclef-2025
Rank: #13
Source: https://www.kaggle.com/c/birdclef-2025/discussion/583457

Thanks so much for the host and I am very delighted to join the competition. Please let me show big thanks for RihanPiggy [https://www.kaggle.com/honglihang](url) without his code we cannot get such goods results. In addition,  Koki (train subset models to boost final solution +0.01+) and Zhang (post-process boost solution +0.004) also made a great effects on the final results. 

# Overview of our solution

This year competation is very common problem in industry, you have relatively clean data in training, but there are very dirty data in your inference stage. How to overcome the domain shifting is the key to win the competation. The train_soundscapes would help us to decrease the gap between the training data distribution and test data distribution.

Our solution can be divided into four stages alomost like all other winners teams:

# Step 1:
Use 2023 2nd code to train the basic SED model with v2s (lb 0.84 ~ 0.85)
remove the humance voice which need to recalculate the duration (lb + 0.003)
use sumix instead of mixup on raw audio singal (lb +0.01)
Koki cleaned the trainin data (lb +0.003)

After all the above steps, the base model can be 0.865 in public which was the goods start point

# Step2:
Use the step 1 model to do the pseudo data and inference the data 5s.
And above 5s clip by random sample into train audios it will greatly boost the lb (+0.03 ~ 0.04)

# Step3:
After finished the above experiments, we tried to find other models to ensemble which was most time-cost. We tried lots of backbone, but none of them was better then v2s. we use the following models:

seresnext26t lb 0.899
v2_b3 lb 0.901

the reason why v2_b3 can work is v2_b3 and v2s have very similar model structure. It can also lead the model ensemble can not give little boost lb (+0.04). I think most of the teams had the same problems

# Step4:
Inspired by 24 top 6 solution, we trained rare specials model and add them into final solution it give us lost of boost õn lb (+0.1)

That's all of our soltuons hope you enjoy. 
Happy kaggle.
