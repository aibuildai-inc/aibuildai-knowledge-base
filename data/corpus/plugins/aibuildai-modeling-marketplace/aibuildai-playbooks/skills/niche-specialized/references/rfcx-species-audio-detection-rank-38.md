# 38 place writeup(image classification on TPU/Colab only with s3 trick)

Competition: rfcx-species-audio-detection
Rank: #38
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220386

Hi all.
First, thanks for a great competition. Thats my favorite type of competitions where there are incomplete or noisy labels and you have to think how to deal with them.
Second, thanks to those who shared code. In particular to the authors of the following kernels:
1. https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2 - that was a great starter and I was just doing edits of that kernel to move on
2. https://www.kaggle.com/aikhmelnytskyy/resnet-tpu-on-colab-and-kaggle - that showed how you can train on colab as well

My code is in the following notebook - https://www.kaggle.com/vzaguskin/training-rfcx-tensorflow-tpu-effnet-b2-with-fp. The final submission is a merge of several versions of submissions from that code(and similar code on colab) plus s3 trick. 

Now, how I got from initial 80+ in the starter notebook to 90+ and silver zone.
1. 5-second cut worked better then initial 10 second
2. Added FP data with masked BCE/BCEFocal loss. I simply calculate BCE loss only on the label I know is missing(and just usual BCE with label smoothing 0.2/Usual BCEFocal on TP data)
3. Use heavier model(B4)
4. Added mixup/cutmix

The best version of that code got .89+ on private LB. 
Than ensembling goes - I just collect all the well scoring submissions(.87+ public) and average them. Ranking average seem to work slightly better than simple average(by 0.001 approximately).
The best private score I could get with that approach is .912

My version of s3 trick is that I multiply s3 by 2.5 and s7 by 2. That gave me .93 on private LB(.918 public). The version I selected for final had same .918 public and .929 private which is pretty much the same.

Again, thanks a lot for the competition. Learned many things and had a lot of fun.

**Upd:** I've added postprocessing from Chris and now [this kernel](https://www.kaggle.com/vzaguskin/training-rfcx-tensorflow-tpu-effnet-b2-with-fp) scores 0.95467 private (gold zone) - which means complete training and inference on Kaggle TPU only within less than 3 hours and gold level score.
