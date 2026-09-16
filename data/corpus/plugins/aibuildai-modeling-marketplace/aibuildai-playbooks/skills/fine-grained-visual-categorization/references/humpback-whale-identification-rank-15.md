# 15-th plcae solution: sphereface, image alignment and multi-layer fusion

Competition: humpback-whale-identification
Rank: #15
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82361#latest-482500

First， thank my teammates for their helps (hanhan chen, zhangboshen, T-mac, zouhongwei) and sorry to them for what happened. We solve this task via classification with angular margin loss. It seems this task is more a fitting competition and the risk of over-fitting is not high.
##preprocess
We oversample images in 5004 class that occur less than 20 to 20. Images are resized to 384 and 512 as Input. Data augmentation is randompadding+randomcrop. We also tried random erase and random affine. But they work not so well as randompadding+randomcrop.
##sphereface
At the beginning, we using pure softmax to classification 5005 class. The best result we obtain is around 0.86X using seresnext50. Then we resort to sphereface. To use sphereface, we abandon new whales, which means we only use around 19K images. This gives us 0.920 using seresnext-50(multi-layer fusion, 384*384), 0.919 using resnext50(multi-layer fusion,384*384). We also tried arcface, which gives us 0.911 using seresnext-50(multi-layer fusion,384*384).
##multi-layer fusion
Actually, we resort to this tick quit early when we used the softmax as loss fuction. In particular, we cat feature from layer 1 to layer 4. In resnext, it gives us feature of 3840 d. After using multi-layer fusion, the public score improved from 0.79X to 0.86X. So I didn't test what if remove it from our best model. However, we tried the original Inception and densenet without multi-layer fusion. Each gives me 0.83 and 0.88 using spereface. So I think this trick really works.
##Image alignment
We trained a self-designed keypoint detector on 1000 Hand-Annotated Humpback Whale Fluke Keypoints dataset provided by Paul Johnson in discussion. Then we affine the image to prefined coordinate using the keypoints by learning a transfer matrix. After using this, we obtain 0.943, which is also our best single model. I will give some example aligned images latter.
##ensemble
Since there is some badly aligned images. We merge models that trained without alignment. And ensemble models trained on 384 and 512. We arrive 0.953.
##Pseudo-Labelling and making use of newwhale
We use our best model to give label to the playground images and use threshold to merge the two dataset. We also tried to making use of newwhale. We designed a loss function to repress the response of newwhale on the 5004 classes. By these tricks, we obtain 0.954 after ensemble.


Really hope I can win a gold next time :)
