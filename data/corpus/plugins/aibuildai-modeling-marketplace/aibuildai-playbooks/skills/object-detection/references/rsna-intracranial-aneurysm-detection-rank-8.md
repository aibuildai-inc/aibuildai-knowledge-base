# 8th Place Solution

Competition: rsna-intracranial-aneurysm-detection
Rank: #8
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution

## Approach:
- Two stage approach to find aneurysms in sequences of MR/CT images without using resampling, 3D approaches or segmentation. 
- Approach the problem as outlier prediction task: we have only a small number of aneurysm spots in the training sequences, but can sample a huge number of negative spots, where no aneurysm is present.
- **Stage-1:** Train a classifier on single images.
- **Stage-2:** Train a Transformer to classify the whole sequence of images based on the extracted features of Stage-1.

## Stage 1:
Based on the given **train_localizers.csv** sample positive (aneurysm present) and negative (no aneurysm present) cases and train a simple image classifier. Use the surrounding images as R and B channel of the image. Two different step sizes are used for stacking. Step 2 simply means not stacking three consecutive frames but instead leaving a gap and taking the next frame. As backbone a Convnext Base Dinov3 is used with a simple classification head. 

[stage-1]

### **Augmentations:**
```
valid_transforms = A.Compose([
                              A.CenterCrop(448, 448),
                              ]) 


train_transforms = A.Compose([
                              A.ShiftScaleRotate(rotate_limit=(-5, 5), p=0.5),
                              A.RandomCrop(448, 448),
                              A.RandomRotate90(p=1.0),
                              A.OneOf([
                                        A.GridDropout(ratio=0.4, p=0.5),
                                        A.CoarseDropout(max_holes=25,
                                                        max_height=int(0.2*448),
                                                        max_width=int(0.2*448),
                                                        min_holes=10,
                                                        min_height=int(0.1*448),
                                                        min_width=int(0.1*448),
                                                        p=0.5),
                                        A.GridDistortion(p=1.0),
                                        ], p=0.5),
                              ])

```

Images are getting resized to **512x512** during pre-processing. Rotation invariant training and in addition a left to right flip by simultaneously flipping the labels is applied. 

### **Sampling:**
To sample negative spots for training of Stage-1, image from sequences without an aneurysm but also images from sequences with aneurysms are used. In the second case, spots where an aneurysm is present are excluded with some boarders. Furthermore, based on the OOF predictions of an early trained classifier, spots where the model predicts false positives are sampled more often. 

[sampling]

## Stage 2:
Based on the extracted features of Stage-1 a Transformer is trained on the complete sequence per UID. For augmentation in this stage, sequences with step-1 and step-2 are extracted and used during training of Stage-2. 

[stage2]

## Ensemble:
- An ensemble of 4 Convnext Base models with two different step sizes is used.
- On sequences > 192 images, only every n image is selected to shrink sequence length (n is calculated dynamically based on original sequence length).
- To shrink the sequence length is only necessary because of the 12h time limit and the incredible crappy Kaggle hardware with totally random runtime based on the assigned server. 
- Nearly half of my submissions are simply timeouts, sorry Kaggle, but it can’t be that the same code without changes hit the 12h time limit or finishes in less than 8h.

## Links:
[Training](https://github.com/KonradHabel/rsna)

[Inference](https://www.kaggle.com/code/khabel/rsna-dual-gpu-ensemble-2step)

[Weights](https://www.kaggle.com/datasets/khabel/rsna-2025-weights-4xbase)
