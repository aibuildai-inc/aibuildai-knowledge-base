# 12th place solution

Competition: czii-cryo-et-object-identification
Rank: #12
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561607

I would like to express my gratitude to kaggle and czii for organizing this competition. I have learned so much from this experience.

---

# summary
I performed segmentation using a 3d unet with the predefined patch size.
My main focus was on enhancing diversity through the following ensemble methods.
- various model types
- various patch sizes
- various tta combinations

### various model types
I used three types of encoders.

1. type1 : 3d encoder (converted from timm resnet18d) + 3d decoder(from scratch)
2. type2 : 2d encoder(timm pretrained resnet18d) + 3d decoder(from scratch)
3. type3 : 3d encoder([csn](https://arxiv.org/abs/1904.02811)) + 3d decoder(from scratch)

### various patch sizes
I used four patch sizes.

1. (64, 256, 256)
2. (32, 352, 352)
3. (32, 224, 224)
4. (32, 128, 128)

### various tta combinations
I applied different tta combinations for each model due to inference time constraints. for example,

1. model a : transpose + hflip
2. model b : rotate90 + rotate270
...

---

# final submission
the final submission was made using an ensemble of 10 models.
|model|patch size|model type|
|---|---|---|
|model1|(64, 256, 256)|type1|
|model2|(64, 256, 256)|type2|
|model3|(64, 256, 256)|type3|
|model4|(32, 352, 352)|type1|
|model5|(32, 352, 352)|type2|
|model6|(32, 224, 224)|type1|
|model7|(32, 224, 224)|type2|
|model8|(32, 224, 224)|type3|
|model9|(32, 128, 128)|type1|
|model10|(32, 128, 128)|type2|

---

# training
- label : radius * 0.5
- loss : dice, focal
- augmentation : flip, contrast, brightness, rotate, mixup, various filters(denoised, wbp, ...)
- regularization : drop path, weight decay
- train data : use all for final submission

---

# not worked
- 2 stage approach
- bigger encoder(e.g. convnext_tiny, maxvit_tiny)
- simulated data

---

# code
https://github.com/siwooyong/CZII-CryoET-Object-Identification

---
