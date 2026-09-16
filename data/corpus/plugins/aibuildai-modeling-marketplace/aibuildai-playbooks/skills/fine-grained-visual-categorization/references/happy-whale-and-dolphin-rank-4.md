# 4th place solution

Competition: happy-whale-and-dolphin
Rank: #4
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320040

Congrats to all the winners. It has been a tough competition. The top 2 teams had a very strong last push. Thanks for the great collaboration once again, my long time teammate @haqishen 

# TL;DR
- Yolov5 Detection
- Dynamic Margin ArcFace with DOLG CNN backbone
 - Best backbone is ConvNext
- Two heads: predicting ids and species
- 3 rounds of pseudo labeling
- Tune new_individual threshold separately for each species

# Detection
We trained yolov5m with image size 512 for 40 epochs. We started by using this [public notebook's](https://www.kaggle.com/datasets/awsaf49/happywhale-boundingbox-yolov5-dataset) predictions as labels. Then visualize examples with low OOF confidence. If the predicted bbox are wrong, remove this example from training set, or fix the ground truth bbox. 

We iterate this for 9 rounds. In the end, most OOF predictions look correct. The OOF iou = 0.93863. But some test set prediction bboxs are still off. In hindsight, we probably should start from scratch and label thousands of images, like other top teams did.

After getting the predicted bbox, we extend it by 20% and feed the crop to arcface model.

# Model
The modeling part is heavily influenced by the recent landmarks competitions. The architecture is Dynamic Margin ArcFace with DOLG CNN backbone. The dynamic margin arcface was introduced by us in last year's Landmark, see detail [here](https://www.kaggle.com/competitions/landmark-recognition-2020/discussion/187757). The DOLG was introduced to the Kaggle community by @christofhenkel in this year's Landmark, see detail [here](https://www.kaggle.com/competitions/landmark-retrieval-2021/discussion/277099).

Other components of the top landmark solutions didn't work here though, including sub center arcface, and vision transformers. All the vision transformers underperform CNNs. The best CNN in our solution is ConvNext.

The only twist in the model is that we output two heads, predicting both individual_id and species.

For augmentations we used the following plus mixup
```
    A.HorizontalFlip(p=0.5),
    A.RandomContrast(limit=0.2, p=0.75),
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.3, rotate_limit=10, border_mode=0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
```

# Training steps
1. Tune everything on 5 fold. i.e. train on 80% of training data per fold, "sacrificing" some individual_ids in order to do cross validation.
2. Ensemble 9 fold models to make pseudo labels.
3. Train 6 best models on 100% training data + pseudo labeled test data
4. Make new pseudo labels from step 3 ensemble
5. Repeat step 3-4 for two more rounds
6. Ensemble the 12 models from last two rounds.

Average single model's public LB in 3 pseudo label rounds are 0.855, 0.862, 0.865

# Ensemble
The best performing backbone is ConvNext. We also trained 3 non-ConvNext for diversity. The final six models are ConvNext Base, Large, XLarge, EfficientNet B7, V2L, NFNet L2. We used diversified image sizes for different models: 640, 672, 704, 736, 768, 800, 832, 864, 896, 960, 1024

The ensemble is done by concatenating single models feature, before computing cosine similarity.

# new_individual threshold
Since final models are trained on 100% of training data, there's no validation scores. New_individual's thresholds therefore need to be tuned on the LB. 

We noticed that different species have different levels of difficulty to predict, hence the optimal new_individual thresholds are different for different species. Tuning each species on the LB is riskly, so we: (1) tuned each species' *relative* threshold separately on CV, (2) tuned overall threshold level on LB, (3) adjust the LB-optimal overall threshold level by CV-optimal species relative thresholds.

This adjustment boosted our public LB from 0.882 or 0.883 to 0.888

We used an ensemble of 15 models’ species head to predict test set's species.

# Things that didn't work
- Vision Transformers
- Reranking Post processing
- Using the previous humpback whale competition's data to pretrain
- Use HFlip to double the number of individual_ids, a trick that worked well in previous humpback whale competition
