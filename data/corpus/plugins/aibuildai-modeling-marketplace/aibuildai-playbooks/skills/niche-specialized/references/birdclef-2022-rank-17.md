# Simple 17th Place Solution [0.81 Public, 0.77 Private]

Competition: birdclef-2022
Rank: #17
Source: https://www.kaggle.com/c/birdclef-2022/discussion/326933

Thanks to Kaggle, competition hosts, and fellow competitors for this very interesting competition. We joined this competition in the last month and had to work hard to understand the competition as neither @neomaoro or I have done anything with audio before this. Both @neomaoro and I worked equally hard on this competition.

**TLDR**

Our solution is based heavily on @philippsinger @christofhenkel and @ilu000's [2nd place solution ](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463) from last year's birdclef competition. We made some modifications to this pipeline, the most significant being resizing spectrograms to 256 * 512 which gave a boost of 0.01 in public and private leaderboard.

**Submission Notebook**
https://www.kaggle.com/code/vexxingbanana/18th-place-solution-0-77-private-0-81-public

**Code Pipeline and Data Setup**

We heavily based our notebooks and python files for training and submitting on @julian3833's [training notebook](https://www.kaggle.com/code/julian3833/birdclef-21-2nd-place-model-train-0-66) and [inference notebook](https://www.kaggle.com/code/julian3833/birdclef-21-2nd-place-model-submit-0-66). We setup training in the cloud on 2 A40 GPUs.

**Bird Classifier**

Our models were very similar and trained the same as last year's 2nd place solution with 30 second clips of train_audio resized into 6 x 5 second parts. We also used primary and secondary labels as targets. For inference, we fed in 6 x 5 second snippets into the model.

We used the following backbones: eca_nfnet_l0, eca_nfnet_l1, tf_efficientnetv2_m_in21k, seresnext50_32x4d, and resnest50d_4s2x40d. Eca_nfnet_l0 and eca_nfnet_l1 achieved the best private leaderboard scores with 5 fold scores of 0.76 and 0.75 respectively. 

During training, we started with the 2nd place solution's training strategy then added some slight modifications. The modifications are as follows:

- Epochs: 20 -> 25
- Optimizer: Adam -> AdamW with learning rate 1e-3 and weight decay 1e-6
- **Resizing: After making a spectrogram, the spectrogram was resized to 256 x 512 using torchvision.**
- Background noise, label smoothing removed.

**Validation**

We couldn't find a way to make a good validation pipeline based on soundscapes since there were no soundscapes with labels given for this competition. Therefore, we simply used validation loss and public leaderboard as validation. 

**Ensembling**

The ensembling of our models was pretty straightforward. We trained 5 fold models of each backbone and then averaged all of the models' predictions together. We tried out a few other ensembling methods such as purely voting and a combination of voting and averaging but found averaging all models to be the best performing. 

**What Did Not Work**

Throughout the competition, we had ups and downs, many successes and some things that didn't work well. Here are some experiments which didn't work out for us:

- Applying PCEN
- Using 2021 Data  
- Attention head
- Adding augmentations such as pink noise, gaussian noise, etc.
- SED models
- Adjusting mel spectrograms values to increase their size (window_size 1024, hop_size 320)
- Using recordings with either rating 0 or ratings >= 2 and label smoothing of 0.01
- More epochs
