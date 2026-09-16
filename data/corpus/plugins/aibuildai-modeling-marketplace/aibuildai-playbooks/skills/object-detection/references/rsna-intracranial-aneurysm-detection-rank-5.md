# 5th place solution with code

Competition: rsna-intracranial-aneurysm-detection
Rank: #5
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution

Thanks to Kaggle and RSNA for hosting this exciting competition. 
It was a great learning experience and it was very interesting to see how much of my computer vision experience could also be applied to medical imaging. I’m quite disappointed with the result, but I see it as an opportunity to learn from other teams.

**Github code**: https://github.com/hoanghuyen797/RSNA-Intracranial-Aneurysm-Detection
**Inference notebook**: https://www.kaggle.com/code/longb173/rsna-iad-final-nb?scriptVersionId=266774552
**Demo notebook**: https://github.com/hoanghuyen797/RSNA-Intracranial-Aneurysm-Detection/blob/main/src/demo-test/test.ipynb

#Overall  Pipeline

## 1. 2.5D image 
For each slice t, I combine slices t-1 and t+1 to create a 3-channel image corresponding to [t-1, t, t+1]

## 2. Exp0: Aneurysm detection
In my experience, using only the classification labels (train.csv) is not as accurate as combining the classification and localization labels (train_localizers.csv). Therefore, based on the labels provided in train_localizers.csv by the host, for each aneurysm centroid, I searched within ±10 neighboring slices and manually annotate bounding boxes for the aneurysm using [LabelImg](https://github.com/HumanSignal/labelImg). 
This process does not require specialized medical knowledge since the aneurysm centroids are already provided.
2 classes: aneurysm (modality CTA, MRA, MRI T1post) and aneurysm_mri_t2 (modality MRI T2)

Then I train 5 models (5 folds) using YOLOv11x 1280
metric   | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4
:-------:|:------:|:------:|:------:|:------:|:------:
mAP50    | 0.705  | 0.647  | 0.766  | 0.702  | 0.691
mAP50-95 | 0.460  | 0.429  | 0.504  | 0.482  | 0.449

## 3. Exp1: Brain detection
For each SeriesInstanceUID in the training set, I generate a single image by averaging all slices. 
Then, I manually annotate the brain bounding box as the following 2 classes: brain (brain in axial view) and abnormal (brain in other views). Each slice in the series will be cropped according to the bounding box predicted by this model. This reduces background noise (especially for slices containing lung regions ...), which improves the model’s accuracy by about **0.03-0.05**.

Then I train 1 model using yolov5n 640: mAP50-95 = 0.948

## 4. Image augmentation
`train_transform = albu.Compose([
    albu.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.5, 1.0), ratio=(0.75, 1.3333), p=1),
    albu.ShiftScaleRotate(rotate_limit=15, border_mode=0, p=0.5),
    albu.OneOf([
        albu.MotionBlur(blur_limit=5),
        albu.MedianBlur(blur_limit=5),
        albu.GaussianBlur(blur_limit=5),
        albu.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.5),
    albu.CLAHE(clip_limit=4.0, p=0.5),
    albu.HueSaturationValue(p=0.5),
    albu.RandomBrightnessContrast(p=0.5),

val_transform = albu.Compose([
    albu.Resize(self.image_size, self.image_size),
])`

Horizontal Flip: It may sound unreasonable, but I applied horizontal flipping to the images and adjusted the labels as following:
Left Infraclinoid Internal Carotid Artery <-> Right Infraclinoid Internal Carotid Artery
Left Supraclinoid Internal Carotid Artery <->Right Supraclinoid Internal Carotid Artery
Left Middle Cerebral Artery <->Right Middle Cerebral Artery
Left Anterior Cerebral Artery <->Right Anterior Cerebral Artery
Left Posterior Communicating Artery <->Right Posterior Communicating Artery
This worked, and my model’s accuracy improved by about 0.01

## 5. Exp2: 2 classification models
2 multi-label classification models (14 classes), image size=384, trained on this competition dataset. For negative series I use all slices, while for positive series I only use slices that contain aneurysm boxes (as reviewed in section 2). The label of each slice is the same as the series label in the train.csv
vit large 384: OOF AUC = 0.8503
eva large 384: OOF AUC = 0.8551
Due to time constraints, I couldn't use 5 models (5 folds) for prediction. I could only use a single model trained on almost the full dataset, using only 50 series for evaluation, and selecting the best epoch based on the public leaderboard.

## 6. Exp3: Multi-task classification + segmentation
1 model (mit b4 fpn), image size = 384, trained on the RSNA dataset. Similar to exp2, for negative series I use all slices, while for positive series I only use slices that contain aneurysm boxes (as reviewed in section 2). The label of each slice is the same as the series label in the train.csv. For the submission, I only used the prediction from the classification task. To create masks for the segmentation task, I use the aneurysm bounding box (2). 
mit-b4 FPN : OOF AUC = 0.8549

## 7. External dataset 
I use the competition dataset and 2 external datasets:
Lausanne_TOFMRA: https://openneuro.org/datasets/ds003949/versions/1.0.1 
Royal_Brisbane_TOFMRA: https://openneuro.org/datasets/ds005096/versions/1.0.3
For the external data, I generate the series label using prediction of 2 models (vit large exp2 + mit b4 exp3), and generate the localization label using prediction of model exp0-aneurysm detection  

## 8. Clean trainset
For negative series in the trainset with 'Aneurysm Present' prediction score of 2 models (vit large exp2 + mit b4 exp3) > 0.9, I will change it to positive and use model exp0-aneurysm detection to create localization label

## 9. Exp4: 2 classification models
2 classification models trained on the cleaned RSNA dataset (8) and external dataset (Lausanne_TOFMRA + Royal_Brisbane_TOFMRA) pseudo labeling (7)
vit large 384: OOF AUC = 0.8558
eva large 384: OOF AUC = 0.8579

## 10: Exp5:  Multi-task classification + segmentation
1 model (mit b4 fpn), image size = 384, trained on the cleaned RSNA dataset (8) and external dataset (Lausanne_TOFMRA + Royal_Brisbane_TOFMRA) pseudo labeling (7)
mit-b4 FPN 384: OOF AUC = 0.8629

## 11. Final submission
The final submission is the ensemble of 6 models:
- Final1: 0.25 exp3_mit_b4 + 0.25 exp5_mit_b4 + 0.125 exp2_vit_large + 0.125 exp4_vit_large + 0.125 exp2_eva_large + 0.125 exp2_eva_large
- Final2: 0.5 exp3_mit_b4 + 0.125 exp2_vit_large + 0.125 exp4_vit_large + 0.125 exp2_eva_large + 0.125 exp2_eva_large

Final1: OOF AUC = 0.8823, public LB = 0.89
Final2: OOF AUC = 0.8767, public LB = 0.89
