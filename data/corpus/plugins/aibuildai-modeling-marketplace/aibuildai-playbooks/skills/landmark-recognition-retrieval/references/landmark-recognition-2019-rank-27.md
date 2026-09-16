# 27th place solution

Competition: landmark-recognition-2019
Rank: #27
Source: https://www.kaggle.com/c/landmark-recognition-2019/discussion/94486#latest-543792

First of all congratulations to all gold and other medal winners. Also thanks to Kaggle/Google for hosting such an extremely challenging and interresting competition.

I'll share a brief overview of the solution I used to get to a 27th place. I'll share my code later when I have some spare time to clean it up.

I downloaded and resized all train and test images to 224 * 224 pixels. Using the pretrained models generated from the Places365 dataset I made a top-5 class prediction for all train and test images. Initially I started with a simple voting classification. For all train images I removed the ones that had 4 of 5 classes predicted as indoor (which can be loosely interpreted as non-landmark). For the test images I used a 3 out of 5 prediction to specify which ones where non-landmark. This removed several hundreds of thousands of images from the train set and also allowed me to specify tens of thousands of images from the test set as non-landmark.

As model I used a pretrained Resnet50 with a custom head layer. I added a Generalized-Mean (GeM) pooling layer followed with a fully connected layer to trim down the number of parameters. I used basic augmentation for all images (flip lr, updown, shear and zoom).  Head-only training I usually did 4 to 8 epochs (each epoch with 1/4 of the train images). Learning rate 0.0005 and a batch size of 128. Full training of the network for 6 to 18 epochs (each epoch with 1/6 of the train images). For full training batch size would be about 20 to 24 (depending on the amount of classes selected). Learning rate would drop from 0.0001 to 0.00003.

Initially I started with using validation of about 10% of the train data. I noticed quickly that it took a lot of additional time and there was no noticable correlation between local metrics and the LB. So I just skipped validation completely...not something you would do in any production scenario but it worked OK for me in this competition.

For generating my submissions I usually used 3 to 4 models from different epochs to create the predictions. I used TTA varying between 2 to 14. For 2 just a simple fliplr of the test image. When using 14 I also made several crops. In the end the simple fliplr seemed to give the most stable results surprisingly.

In the last week I spent a couple of evenings to further optimize the landmark/non-landmark identification and apply that to the stage2 test images and also to further select train images.
That way I was able to gain a 50% increase in my score in the last 3 days. The final model was trained on 83K of classes.

Things that didn't work for me:
I tried various Xception models. Training took however way longer then my Resnet50 models. Also I got the impression that Xception did not handle the large amount of classes very well. In the end I tried 4 or 5 different models but decided to stick with Resnet50.
I also tried using the Hadamard classifier to be able to cope with the large amount of classes. The performance seemed quitte good but I was only able to get a model with a significant score on the LB if the number of classes that I used was not to large (&lt; 10K ). 

I'am looking forward to reading the Gold solutions from the winners.
