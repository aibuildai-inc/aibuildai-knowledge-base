# [Private LB 20th]How did I reach 0.82 from 0.68 on the public LB?

Competition: hubmap-organ-segmentation
Rank: #20
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354590

I'm very inexperienced because It's my first time to compete as a kaggler, and I have learnt so  much from this competition! I will share some tricks which really work in my experiments. Following sores mentioned are reached by single model(5 folds).
## Expriments process
### 0.68
**backbone**: resnext-50
**image size**: (256,256)
**tile**: True
**threshold**: kidney=0.225,
    prostate=0.225,
    largeintestine=0.225,
    spleen=0.225,
    lung=0.225
**cutmix**: True
We just used unext without any other tricks. And we tiled the original size into 1024 and then resized 1024 to 256.
### 0.76
Thanks for the sharing  from [https://www.kaggle.com/hengck23](url) who really helped us.
**backbone**: mit-b2
**image size**: (768,768)
**tile** False
**threshold**: kidney=0.225,
    prostate=0.225,
    largeintestine=0.225,
    spleen=0.225,
    lung=0.225
### 0.79 
**backbone**: mit-b2
**image size**: (768,768)
**tile** False
**threshold**: kidney=0.225,
    prostate=0.225,
    largeintestine=0.225,
    spleen=0.1,
    **lung**=0.1
**cutout**: True
cutout can help accelerate Training and  improve your scores! A better threshold can bring very much improvement.
### 0.80 
**backbone**: mit-b2
**image size**: (768,768)
**tile** False
**threshold**:  kidney=0.225,
    prostate=0.225,
    largeintestine=0.225,
    spleen=0.1,
    lung=0.1
**cutout**: True
**multiple initialization**: True
**stain tools**: True
Multiple initialization is helpful to model ensembling!

    init = [xavier_uniform_init,xavier_normal_init,he_init,kiming_init,orthogonal_init]
    model.aux.apply(init[fold%len(init)])
    model.head.apply(init[fold%len(init)])
    model.logit.apply(init[fold%len(init)])
### 0.81
**backbone**: mit-b2
**image size**: (768,768)
**tile** False
**threshold**:  kidney=0.225,
    prostate=0.225,
    largeintestine=0.225,
    spleen=0.1,
    lung=0.1
**cutout**: True
**multiple initialization**: True
**pseudo label**: True
**stain tools**: True
We only use the pseudo label for lung while using the pseudo label for other organs will be harmful to model.
### 0.82
**backbone**: coat
**image size**: (1024,1024)
**tile** False
**threshold**:  kidney=0.45,
    prostate=0.45,
    largeintestine=0.5,
    spleen=0.2,
    lung=0.1
**cutout**: True
**multiple initialization**: True
**pseudo label**: True
**stain tools**: True
Thanks for the sharing from [https://www.kaggle.com/hengck23](url) again!
### Other experiments
We have spent too much time experimenting on cutmix which means we paste the label copied from another sample so that we don't have enough time to do more helpful experiments just like I have mentioned. We are anxious that cutmix will work well but the truth is that it is harmful to training. However, we used self-cutmix for model ensemble and the probability is 100% which means some models are trained on very different data. In this way, the harm can be controlled and it's helpful for ensembling.
#### I really want to know how to reach 0.83 or 0.84? Select the best models from dozens models you have trained? Find better thresholds for different organs? Larger image size like 1536?
Updating...............
