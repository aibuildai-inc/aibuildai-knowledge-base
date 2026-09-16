# 26th place solution

Competition: intel-mobileodt-cervical-cancer-screening
Rank: #26
Source: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/35111

First thanks to the organizers and the congratulations to winners, and those who are satisfied with their leaderboard placement and/or effort! Thanks to my team mate! As usual I learned a lot.

Team OdiNN formed just hours before the team merger deadline. We (Graham and me) had a short conversation before the deadline and agreed to team up.

We then started to discuss our individual models, and we realized that we had two totally different approaches to a solutions. Graham had used ROI on the train images and fully trained neural networks with VGG16 and VGG19 structures and then averaging over different seeds.However I had used train and additional dataset and done transfer learning from Keras imagenet pretrained neural networks. (vgg16, vgg19, resnet50, xception, inception_v3). The I trained many models and stacked them.

Since we had this totally different approaches and had not agreed on any common train/test split or sharing a common seed, we basically made the following masterplan: We take the each of our best models (Grahams best and my best) and then we simply average the two based on Stage1 leaderboard feedback.

My stacking model.
I extracted the features for all images (train and additional) through pretrained vgg16, vgg19, resnet50, xception and inception_v3. Each image was flipped and rotate to give me 8 different flips and rotations. Each of these feature set was then trained in 5 fold cross validation in six different learning algorithms. (I safely kept flips and rotations of one image within the same fold to avoid leakage over the folds) The six learning algorithms where: KNN, XGBoost, Random Forest, ExtraTrees, Logistic Regression and fully connected neural network. I kept the same hyperparameters for these learners for all feature sets. This then became 30 models (5x6) where each was trained in 5 fold cv. Out-of-fold predictions where taken and saved of course.

These 30 models where then stacked to a three second level models. An XGBoost model, Logistic Regression and another fully connected neural network. 

At the third level I tried to do weighted geometric averaging, however it then appeared to me that the there must have been a leak over my CV folds. I guess some of the images are actually from the same patient. :-(  So, the weighted geometric (nor arithmetic) averaging did not work and I ended up with the plain unweighted arithmetic mean. I didn't notice CV-fold leak until the second day of stage two, and training the first stage took about 50 hours, so I had no chance of solving this problem.

The other model that was blended with my stacked model was based on ROI extraction of the images and then fully retrain neural networks of VGG16 and VGG19 neural network. He only used the original train set (not the additional data). Graham may elaborate on this. If I understand him correctly he is averaging each over five training sessions with different seeds. The problem he got was that the vgg19 neural net did not manage to make it to the deadline so one vgg19 net was used in the final mix.

Thanks all!
