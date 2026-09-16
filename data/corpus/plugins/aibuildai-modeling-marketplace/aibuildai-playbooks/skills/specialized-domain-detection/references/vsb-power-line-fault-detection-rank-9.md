# 9th Place Solution Overview

Competition: vsb-power-line-fault-detection
Rank: #9
Source: https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/85258#latest-499162

Congratulations to all the winners! 
And thanks a lot to VSB/Enet Center and Kaggle for this exciting competition. Since this is my first Kaggle competition, I'm surprised and glad to this honorable result.<br><br>
 
I'm going to briefly share all of you my 10th place solution, which keeps the score of 0.688 in the private LB. <br><br>


First of all, I have to tell you that this solution is not my best public LB model (RNN), which is in the end 13th place in public LB (0.766).  
I think that the keys of this competition are high-pass filtering and DWT-denoising. In this point, I owe a lot to Jack's [kernel](https://www.kaggle.com/jackvial/dwt-signal-denoising).<br><br>

As many other guys mentioned in public kernel and discussion, the problem of this competition is the discrepancy between train / test distributions(and the unstability of CV / LB resulted from that).
So, I chose as a final submission more *conservative* RNN model, which kept modest score in public LB but might be more robust in my view.<br><br>


In order to choose a more robust model, I spent time to decrease val_auc score in adversarial validation.
I'm going to show what I thought and tried as below.<br><br>

1. Without any filtering / denoising, the discrepancy between train / test distributions was huge (val-auc was over 0.96 in adversarial validation. So, I thought that a model without any preprocessing was in danger of shake.
2. With high-pass filtered and DWT-denoised, val-auc decreased to under 0.78. Therefore, I selected high-pass filtered and DWT-denoised model as a base one (high-pass filtering had the stronger influence).
3. Next, I cut down features which contributed to increase val-auc in adversarial validation, and finally succeeded to decrease val_auc to under 0.7. (Notwithstanding, the instability of CV / LB was not completely solved...)<br><br>


The architecture of my RNN model is simple and not novel. 
Similar to other guys, I selected [Bruno-based](https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694) BiLSTM x 2 + Attention model and [Tarun-based](https://www.kaggle.com/tarunpaparaju/vsb-competition-attention-bilstm-with-features) BiLSTM x 2 + Attention + feature concatenation model.<br><br>

And I hard-voted 8 predictions (each resulted from stratified 5- or 4-fold CV models) which were a little bit different from each other in terms of features and random seeds.
When I selected models, I kept in mind to choose ones whose train / validation losses were relatively small(because of CV / LB scores' instability).

Moreover, I used recently released AdaBound optimizer. Although It contributed to increase local CV score, I don't know it is a good choice especially in this easily-overfitting competition.
