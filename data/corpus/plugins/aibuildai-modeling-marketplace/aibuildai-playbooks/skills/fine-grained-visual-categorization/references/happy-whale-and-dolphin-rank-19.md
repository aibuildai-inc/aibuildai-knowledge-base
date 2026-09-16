# 19th Place - Single Model LB 860 Without Pseudo

Competition: happy-whale-and-dolphin
Rank: #19
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320298

# 19th Place HappyWhale Solution
We're very happy to present our 19th place HappyWhale solution. Our team consists of @ragnar123 @mpware @bolkonsky It was a pleasure to work with and learn from all these Kaggle Grandmasters. Thank you Kaggle, Ted Cheeseman, collaborators, and everyone at Happywhale for a great competition!

# Summary
* Large Backbones like **EffNetB7**, Large Image sizes like **768x768**
* Two **ArcFace** modules. One for species and one for individual_id both **m=0.19, s=19**
* Six datasets; Mpware **fullbody**, Mpware **fin**, Jan fullbody, Jan fin, Phalanx detic, Awsaf yolo
* Infer all six datasets and **Bayesian optimize** weighted average of six embeddings
* **Eight fold CV**. Tune `new_individual` threshold on CV. Best single model **CV 0.865 LB 0.859**

# Datasets
We would love to train our models with only the original images and have the model learn to place attention on the dorsal fin and/or fullbody. However the images are very large like 2500x3500. So instead we used six datasets where important parts of the image were cropped and then resized to either 512x512, 640x640, or 768x768.




# Models
* Image Sizes 512x512, 640x640, 768x768
* EffNetV1-B5, EffNetV1-B6, EffNetV1-B7
* EffNetV2-L, EffNetV2-XL
* ConvNext-L
* Batchsize = 64
* TensorFlow trained with **Nvidia 8xV100 GPU** and CoLab TPU

# Train - One Model All Datasets
We train one model on all 6 datasets for 20 epochs exponential learning rate decay with 5 epochs warmup. So the model sees every train image 120 times. We also use data augmentation and two ArcFace heads.


# Infer - Bayesian Optimized Average of Six Embeddings
We infer the model on each dataset separately and get six embeddings for each train and test image. Next we use Bayesian optimization on CV score to find the optimal weighted average of these 6 embeddings to use for KNN matching.


# Ensemble
Our best single model had 8-Fold CV 0.866 and LB 0.859. We trained about a dozen different models and then ensembled the `96 = 12 x 8` fold models by using a voting ensemble of each fold model's 5 predictions. Our ensemble result was LB 0.868.

# How To Improve
After reading other top teams' winning solutions, we believe that the next step to boost our model's LB would be to pseudo label the test images and then retrain our models using train and pseudo labeled test. I believe this can boost LB as much as `+0.010 to +0.020` !

# UPDATE
After the competition ended, I retrained two of our 12 ensemble models using pseudo labels. The public LB boost **+0.009** and the private LB boost **+0.016**. This confirms that pseudo labeling is very powerful in this competition. 

Just adding pseudo to two of our 12 ensemble models boosts our final placement to Gold Medal finish! I assume if we add multiple rounds of pseudo to all our ensemble models, we can climb further into Gold Medal zone!
