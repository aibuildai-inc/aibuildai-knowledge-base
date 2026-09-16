# 21st place solution [LB 0.948] on simplified data only

Competition: quickdraw-doodle-recognition
Rank: #21
Source: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73754

Thanks everybody for the competition! The task was really interesting and had huge amounts of data. Thanks very much to my teammate who writes really perfect code on Python. You should understand me, he writes **comments** to functions and uses **types** in Python! Also thanks to [@scitator][1] for his perfect training ML framework [Catalyst][2], all models were trained with the help of it.
 
Now I’d like to tell you about my and [Artyom Palvelev][3] solution using simplified data only.
#data preprocessing#
Pictures of size 128x128 gave the best combination of score and training time. I don’t have time information, so I encoded the following data in three channels:

 1. The index of line (linearly from 10 for the first line to 255 for the last one)
 2. The number of strokes in the line
 3. Just constant 255
#Our models#
Firstly, I tried models from forum like MobileNetV2, but they showed poor performance, so I trained something deeper. My first good model was SE_ResNext50 0.942 LB. I used CosineAnnealingLR and averaged top4 checkpoints.
Then we merged with Artyom and tried different model architectures. Blend with his models gave 0.944 LB. In the final submission we had:

1. SE_ResNext50 (~0.942LB)
2. SE_ResNext101 (~0.944LB)
3. NASNet-A-Large (~0.944LB)
4. SENet154 (~0.945LB)
5. CBAM_ResNet50 (~0.941LB)

We also used *gradient accumulation* to increase batch size to 1024 because data was noisy and bigger batch size gave better performance.
#LGBM#
We decided to use LGBM to ensemble our models because it is fast and usually gives better results than other methods. We used same idea as [Pavel Ostyakov][4] described in his [5th place Cdiscount solution][5]. We predicted top10 classes by our best network, concatenated probabilities of other networks, added class_id feature and gave binary label: whether the class_id is correct or not. This method resulted in validation score 0.001 higher than a simple average. So, if you have enough time, always try LGBM to ensemble.
Final submit with LGBM -&gt; 0.948 LB
#What we tried and didn’t work#

- CatBoost, RF, XGBoost, ensemble of 3rd level
- Tuning models on clean data. We predicted the train dataset with our best models and dropped out pictures with small probability for the correct class (about 1M samples). It gave small boost on validation, but we didn’t have enough time, so we only trained a couple of models for about 2-3 epochs. So, I suppose, it is also a good idea for noisy data.

#What we didn’t try but it worked#

1. We didn’t noticed that the test is balanced (we could use same technique as in [Camera Model Identification][6] and get gold) Others say, it give plus ~0.7% score to any submission.
2. TTA with deleting 20% strokes (people on forum said it also improved score). I tried just light augmentations like flips and shift_scale_rotate but they made score on validation even worse.


  [1]: https://www.kaggle.com/scitator
  [2]: https://github.com/Scitator/catalyst
  [3]: https://www.kaggle.com/artyomp
  [4]: https://www.kaggle.com/pavelost
  [5]: https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/45733
  [6]: https://www.kaggle.com/c/sp-society-camera-model-identification
