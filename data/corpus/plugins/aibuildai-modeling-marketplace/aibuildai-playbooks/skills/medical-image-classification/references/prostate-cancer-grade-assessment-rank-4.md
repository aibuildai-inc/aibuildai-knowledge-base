# 4th place solution [NS Pathology]

Competition: prostate-cancer-grade-assessment
Rank: #4
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169113

First of all Thank you very much to organizers and thanks to @sinpcw  for fighting with me!
Our solution is simple.
We ended up with the following models in our final ensemble submission.

&gt; efficientnet_b5 x 4
&gt; seresnext101-64x4d x2
&gt; seresnext101-32x4d x2
&gt; resnest101e x2
&gt; gem+efficientnet-b3 x 1

We used hard voting for the ensemble method, not soft voting. This definitely improved our score!
We also used the average if the one getting the most votes in our hard voting did not get more than 1/3 of the total votes. But this method only worked on publicLB.

In training, We use the technique of tiling the images in the following links. This allows us to ensure that the tissues are evenly distributed across all tiles.It is also able to perform Data Augmentation by changing the scaling factor. We use 512x512x16 from middle layer. We've also tried using 1024x1024x16 from highest resolution layer, but there was no improvement.
https://www.kaggle.com/hirune924/image-loader-test

We also use syncBN. This was important when training the larger models.
blue line is normal BN, brown line is syncBN.


### O2U-Net
(This method seemed to work in the privateLB, but we didn't use it in the end because we couldn't see the effect in the publicLB)
We also tried using O2U-Net to remove the data noise, but didn't work in publicLB.
But data cleansing of radboud only seemed to work for privateLB.
seresnext50 trained on noise removed dataset for only radboud achieves 0.933 in privateLB.(if without data cleansing privateLB 0.915)
https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf

I share a notebook that calculates the noise level based on the recorded loss by O2UNet.
https://www.kaggle.com/hirune924/o2unet-loss-aggregate
The effect of data cleansing on private LB is also described in the 1st place solution.
https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169143

### Usefull tools
PyTorch Lightning https://github.com/PyTorchLightning/pytorch-lightning
Hydra https://hydra.cc/
Neptune ai https://neptune.ai/
KAMONOHASHI  https://github.com/KAMONOHASHI
