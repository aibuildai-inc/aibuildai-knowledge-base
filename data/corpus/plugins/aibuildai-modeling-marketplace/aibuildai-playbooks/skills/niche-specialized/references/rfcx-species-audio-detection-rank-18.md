# [completed] rank 19th solution in 13 days - LB 0.937/0.939 (public/private) 5 fold model

Competition: rfcx-species-audio-detection
Rank: #18
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220309

[summary]

- Design a sliding window conv classifier net. Trained with tp and fp annotations (no pesudo label) on resnet34, this architechiture achieved LB 0.937/0.939 (public/private) in 5 fold model. A single fold model gives 0.881/0.888 (public/private). The first max pooling resnet34 is removed to make the stride of the backbone 16 (instead of 32)

- Final submission is LB 0.945/0.943 (public rank 15/private rank 19) is an ensemble of this network with different CNN backbone.

---

I am grateful for Kaggle and the host Rainforest Connection (RFCx) for organizing the competition. 

As a Z by HP & NVidia global datascience ambassador under https://datascience.hp.com/us/en.html.HP & Nvidia has kindly provided a Z8 datascience workstation for my use in this competition. Without this powerful workstation, it would not be possible for me to develop my idea from scratch within 13 days. 

Because I can finish experiments at a very great speed (z8 has ax quadro rtx 8000, NVlink 96GB), I gain a lot of insights into model training and model design when the experimental feedback is almost instantaneous. I want to share these insights with the kaggle community. 

Hence I started another thread to guide and supervise kagglers who wants to improve their training skills. This is targeted to bring them from sliver to gold levels. You can refer to:
https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220379
https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/217238

---
data preprocessing
- log mel spectrogram with modified PCEN denoising (per- channel energy normalization). A 10sec clip has melspec of size 128x938
n_fft           = 2048
win_length = 2048
hop_length = 512  
num_freq   = 128


---
augmentation
- I haven't tried much augmentation yet. For TP annotation, random shift by 0.02 sec. For FP annotation, random shift width of the annotation and then random flip, mixup, cut and paste of the original annotation and its shift version.
- Augmentation is done in the temporal time domain because of my code. (this is not the best solution)
- It can be observed that if I increased the FP augmentation, the bce loss of the validation FP decreases and the public LB score improved.
- heavy dropout in model

---
model and loss
- please see the below images
- during training, I monitor the separate log loss of validation TP and FP. I also use the LRAP assumpting one label (i.e. top-1 correctness). This is to decide when to early stop or decrease learning rate. These are not the best approach.







daily progress:

I enter the competition are reading CPMP post, saying that he has 0.930 LB in the first submission. WOW! Reading between the lines, it means:
1. the public kernels are doing it "wrong". It could be the problem setting, the network architecture, the data, or some magic features. there is something very fundamental that can do better than the public kernel.
2. 0.930 is quite high score (in the gold region). With 2 more week to the deadline, I decided to give it a try. if I solved this unobvious puzzle, i may end up in the gold region as well.
3. the first step is to go through the public kernel (some PANN SED) and the baseline image classifier. I want to see what most people are doing (and could be improved)
4. after working for a week, I realize that there are two major "potential flaws":
  - treating it as a multi-instance learning SED. Well, MIL could be a solution, but the problem is that we don't have bag label (clip label). Most MIL required bag level label, but lack instance level label (segment label).Here we have the opposite.
  - not using FP in train. Most public kernel use only TP as train only. 
5. hence I start to design my own network architecture. The first step is an efficient way to do crop classification based on mean annotation box. Hence I design the slide window network.
6. The next step is to use TP+FP in training
7. The last step is to use pesudo labels, but I don't have time to complete this. But I do have some initial experimental results on this. Top-1 (max over time) pesudo labels is about 95% accurate for LB score of 0.94. This is good enough for distillation.
8. Because pesudo labeling requires many probing to prevent error propagation, I could not do it because I don't have sufficient slots in the last days. Worst still, there is no train data at clip level at all. This makes local testing impossible.

I was able to train 5-fold efficientB0 and resnet50 in the last day. Because of the large GPU card of the HP workstation I am using, I can train large models with large batch size. When I compared training the same model with smaller batch size on my old machines, I find that the results are different and inferior, even if I use gradient accumulation.

I strongly feel that we have reached a new era. I think this is also why the latest RTX3090 has larger GPU memory than the previous cards. The future is the transformers .... meaning more GPU memory. That's is how fast deep learning are moving!
