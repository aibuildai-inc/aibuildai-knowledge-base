# 25th solution overview

Competition: human-protein-atlas-image-classification
Rank: #25
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77276

Data Processing : simple rotate, flip on 512x512/1024x1024 RGBY

Loss : 
I used focal loss and took lots of time to optimize gamma(I used 2 and 2.5 in final version), and it seems alpha=1, gamma=2.5 actually works better on public leader board but not so good on private leader board.

Model : 
I used  se-resnext50 on 512x512/1024x1024 size images as baseline model and resnet34 on 512 size as a low capacity model. I made 5 cross validation models(total 24 models) and weighted averaged them(I gave more weight on gamma=2.5 models). Due to the limitation of time and resources, I couldn't make the full planned ensemble models.

Thoughts:
Although I exhausted to make 24 models as an ensemble and it seemed worked well in public leader board, but my hidden best private score was from the ensemble which I just 'or'ed all output labels of two single models and have relatively low public score,(0.571/0.538),(0.561/0.525) each(※(public LB/private LB))). And It just turned out that it reached private LB 0.550 which was slightly better than my final ensemble model,0.547. They were from resnet34 and se-resnext50 on both 512 size images, so maybe 24 models for ensemble was too much.
