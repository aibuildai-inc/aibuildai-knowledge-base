# 31st place solution completely on kernels

Competition: landmark-recognition-2019
Rank: #31
Source: https://www.kaggle.com/c/landmark-recognition-2019/discussion/95077#latest-551166

Hello

Thought people might be interested to know how to run this entire competition on Kaggle kernels and still finish in good positions on LB. So here is my solution to it. I have completed both the Recognition and Retrieval challenges relying solely on Kaggle kernels. Thank you Kaggle!

P.S. I had earlier authored a public [kernel](https://www.kaggle.com/mayukh18/dataset-exploration-and-simple-on-the-fly-training) in Keras at the start which showed this can be done. Even Artyom did this in Torch (https://www.kaggle.com/artyomp/resnet50-baseline).

### Approach for Kaggle Kernels

The solution was run completely on Kaggle Kernels. Since the dataset size is around 500 GB, taking advantage of the size of each tar file, the training was done on a download-train-discard methodology. In each mini-epoch, the data generators downloaded one tar file from the 500 tar files; extracted it; created batches from the valid images and at the end of the epoch deleted all the files. Each tar file downloaded in about ~30 secs, and a mini-epoch of about ~7000 images took a total of 100-110 secs(including everything).

### Model Architecture

- I used a ResNet50 with weights pretrained on imagenet. Chopped off of the final layer. Added a GlobalMaxPooling layer and a subsequent softmax layer at the end for the output. 
- Used landmarks which have atleast 20 images.(~55k landmarks)
- Resized every image to 192x192 pixels. 
- Used Adam optimizer with a rate of 0.0002 at the staring and reduced it after each ~150 mini-epochs. 
- Trained the model for one full epoch(500 mini-epochs) after which it started overfitting. Could have tuned the hyperparameters better for the subsequent epochs but there was not much time left for experimentation.
- Did not use any external dataset for non landmark images.
- TTA somehow gave me worse results.

Special mention to Dawnbreaker's [solution](https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57152) from 2018 version of the challenge which helped me improve my results.

What didn't work? ResNet101, DenseNet etc. and more. Even ResNet50 with 128x128 images didn't give any good result. I initially started off with a ResNet101 but it failed to match the level of performance of the ResNet50. Taking this from a comment of Dawnbreaker in the above solution thread:
&gt; "Actually I tried ResNet101 and got a higher val accuracy but a worse LB performance (LB:0.101). Eventually I think maybe an accurate prediction for the landmark images is not as important as an efficient rejection for the non-landmark images."

Perhaps this makes sense. Add to the fact that here we were predicting on ~55k classes compared to last year's 15k, rejection seems even more important.

The complete organized code(including the Keras generators) and pretrained model can be found here: (https://github.com/mayukh18/Google-Landmark-Recognition-Retrieval-2019)

Happy if anyone finds this useful. Cheers!
Mayukh
