# 11th place - [ods.ai] Dies irae - solution with code

Competition: siim-acr-pneumothorax-segmentation
Rank: #11
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107518

*from Arthur:*
I dived in somewhere 5 days before the merger deadline. During this time, I was unable to train the model better than 0.81 DICE on local validation. This made me extremely angry and I decided to find out the secret sauce of this task. For this, I teamed up with Vova - he had pretty high score on LB. And I was sure that he had a good solution...

Features of the solution:
1. Loss: BCE + DICE. No weighing or log. Just the sum of the losses.
2. Batch 2. Just because.
3. A mixture of Unet and Linknet with a backbone from se-resnext50, se-resnext101, SENet154.
4. ConvTranspose in decoder.
5. Final Convolution 2x2.
6. No image normalization at all.
7. Sampling in which only batches containing non-zero samples are taken.
8. Scheduling: 40 epoch of Adam with CosineAnnealing, 15 epoch of SGD with CyclicLR, 16 epoch of Adam with CosineAnnealing.

I've tried:
1. Transfer all these "features" to my pipeline
2. Transfer only those that seem reasonable
3. Fix “features” in the original pipeline

All this worsened score.

In the end, my contribution to the final solution was that I rewrote most of the code except the train loop itself. Made it possible to learn parallel in folds on GPUs. And finally I trained all the models on my hardware.


*From Vova:*
I decided to participate in this competition on early middle stage due to some intersection with my professional domain - medical imaging. That was mostly the second time when I saw the lung scans and first time discovering its diseases.
As Arthur described above, pipeline which gave first more or less acceptable results didn't mean anything insane, with quite a lot not really optimal solutions, but digging deeper (and mostly longer) highlighted several points: 
1. Training is essential with small batch size
2. Deep models perform better (at least than 18/34)
3. Input size extemelly important (final models trained on 1024)
4. Sampling of not empty batches (works not the first time specifically with medical imaging)
5. Second stage training with SGD after Adam starting from the previous best - it's way more stable with slight improvement
6. Dilation works well in my case (seems models didn't cover required surface)
7. Removing of small objects
8. Light augmentations - Hflip, RandomRotate90, Transpose, ShiftScaleRotate, ContrastBrightnessGamma
9. Classification didn't work for me
10. TTA only as hflip

After the team merge the problems with computing resources and straight hands for proper coding were solved, and we moved forward to climb on LB.
The final (including most intelligent) tricks in our solution from post processing:
1. Random search for the binarization threshold, small objects removal - one set for all models
2. Dilation which applies in dependence on the TTA cross-agreement (in terms of Dice score) for each model
3. Ensembling via union of the binary predictions with agreement (Dice threshold) between at least 2 models (worked way better here than averaging or voting). Training all final models by the same pipeline allowed to be more or less sure in this strategy.


Logs:
1st Stage:


2ns Stage:


[Code](https://github.com/n01z3/kaggle-pneumothorax-segmentation)

In the end, despite the final standings, we decided to publish this overview before the end of the competition in order to share our approach and ideas.
We would like to thank the sponsors, hosts and all involved in this challenge from both sides - competitors and organizers.
