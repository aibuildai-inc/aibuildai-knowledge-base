# Silver zone solo, no credits used.

Competition: deepfake-detection-challenge
Rank: #44
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145720

I regret I did not join this competition earlier, I had just one month of intensive work. 
This should be my first silver medal. I have read many write ups about getting good score and many of them lack one detail: some guys work pretty hard to climb high - this is very important detail. When I read comments like: "I worked 10 hrs/day on this competition" - I could not believe(maybe it is a bit exaggerated), before that I thought - well, ok, 2-3 hrs/week. This time I quit my job partially because I wanted to join some Kaggle competition and take it seriously plus I had no computer vision experience at my job and DFDC seemed very interesting to me. Finally I can say that I spent 30-40 hrs/week in average on this competition, working also over the weekends. I did not use any external machines for training, I joined late and could not use credits. My PC specs are 32 GB RAM, 128 GB SSD + 256 GB SSD, GeForce GTX 1080 Ti. SSD helped a lot in this competition.

I believe the main "magic thing" for me was proper train-validation technique. For example, I have trained EfficientNetB1 which receives 224x224 image as input and outputs a binary number(fake or real) using the next data organization:
For each epoch:
1. Select single face image from unique real video(random frame).
2. Select random fake video which was originated  from a real video in a way that you *do not take two or more fake videos* originated from one unique real video.  Then you pick the random frame and random face from each fake. Eventually you get the same number of subsampled fakes as for real samples.
3. Every epoch you pick faces/frames/videos randomly.
4. Also you split train/validation set by fixed folder-wise split for all epochs.
1 EPOCH takes just 5-10 minutes(if you have all images extracted on your SSD)! And whole training takes just 1-1.5 hours! Finally I had validation score ~0.22-~0.25 for different models.
I extracted 20 frames during inference, classified them and calculated the median of predicitons.

I trained LRCNN(EFNB1 as backbone) as well, its input is only 3 consequent frames.
My final solution is the average of three single frame models multiply by 0.85 and one LRCNN model multiply by 0.15.

I tried toooooons of things, I trained Increption-ResNet V2 during 2 days, attached the fan to my PC and I even was afraid that I will wake up in the house full of smoke :) I believe I could do better if a day had 50 hours instead of 24, if I had teamed up with someone. This competition has no end, we can apply here infinite number of approaches.

I would like to thank guys like Human Analog and Shangqiu Li - they deserve many upvotes. And thank you all for participating!
