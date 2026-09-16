# 10th place solution

Competition: open-images-2019-object-detection
Rank: #10
Source: https://www.kaggle.com/c/open-images-2019-object-detection/discussion/111266

**TL;DR**: 
- I'm using mmdetection framework and it's really convenient;
- Best single model (cascade rcnn with imagenet pretrained resnext101) + TTA (horizontal flip, multi-scale testing (600, 900), (800, 1200), (1000, 1500), (1200, 1800), (1400,
2100)) achieves 0.499 public;
- Split datasets into 6 subsets by frequency and then finetune on them for 1-2 epochs. --&gt; appr.  0.05+ increase;
- Parent class expansion gives appr. 0.01+ increase;
- Weighted ensemble (from [ZFTurbo's solution last year](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64633#latest-590283)) of all my high score submissions --&gt; 0.608 final public score.

**1. Single model**
At the very beginning, I was going to attend the "Visual Relationship Detection" track. But then I realized that I didn't have a good object detection model for that one. So I started with faster rcnn+resnext101, it takes me about 20 days to train 24 epochs and results in 0.446 on public lb.
Similarly, I trained cascade rcnn+resnext101, cascade rcnn+senet154 for 12 and 8 epochs respectively.
I just leave these models training for several weeks, do my daily work and give up "visual relation detection". 
The best single model is cascade rcnn+resnext101, which was accidently trained for 19 epochs (6 epochs longer than planned). So maybe I should train longer for each model :).
**Conclusion**: my single models are weak. They should be trained longer.

**2. Finetune**
Since the classes are very unbalanced, I split the dataset classes into 6 subsets simply according to frequency and finetune on them using faster rcnn+resnext101 model:
- Classes 0-50, appr. 1411368images, 2 epochs, lr 0.001
- Classes 51-100, appr. 308352 images, 2 epochs, lr 0.001
- Classes 101-200: appr. 208096 images, 2 epochs, lr 0.001
- Classes 201-300, appr. 93140 images, 2epochs, lr 0.001
- Classes 301-400, appr. 45840 images, 1epoch, lr 0.001
- Classes 401-500, appr. 19316 images, 1epoch, lr 0.001
Last two weeks before final deadline, I found one huge bug in my code.
After solving this bug, ensembling the predictions of finetuned models gave appr. 0.05+ increase on public LB.
**Conclusion**: solving class imbalance problem is the key to top silver or gold medal.

**3. TTA and Final Ensemble**
Some teams merged and I may have the chance for solo gold.
So I did the followings:
- TTA for each model: horizontal flip, multi-scale testing with (600, 900), (800, 1200), (1000, 1500), (1200, 1800), (1400, 2100) image size;
- Expand parent class for each prediction after inference.
- Weighted ensemble (from [ZFTurbo's solution last year](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64633#latest-590283)) of all models.

**Other tricks:**
- Increase the number of limited boxes for each image, even though they are of low confidence. I choose 600 as upper limit in my final submission.
- I wasn't able to finish 12 epochs for my cascade rcnn+senet154 model. Nevertheless, ensembling it gives slight improvement.

**Not work for me:**
Soft-NMS: tried to use it for ensemble, wasn't going well. 

**Planned but not implemented:**
- Use mask annotations from the segmentation track;
- Multi-scale training and mixup augmentation.

**Final words:**
Let's play fair. 
Peace&amp;love. 👍
