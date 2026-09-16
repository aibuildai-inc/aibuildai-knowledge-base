# Solution Sharing and Congratulations

Competition: the-nature-conservancy-fisheries-monitoring
Rank: #12
Source: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/31538#175479

My approach is pretty empirical, which is based on my intuition. I hope to develop a more scientific approach to quantify network uncertainty after the competitions.

At the beginning of the competition, I realized that the data set is very small so fine-tuning on existing network weight is necessary. Meanwhile, I find that there are test stage 2 so I need to formulate the problem as an object detection problem to handle unknown situations. I did not use boat ID since I don't think it is the right way to go. Once the problem has been formulated, I read many literature on object detection and found a few candidates in this area. 

I have tried SSD, YOLO, py-rcnn, py-rfcn, pvanet for object detection. The final detection workflow was a combination of py-RFCN(resnet-50 and resnet-101, using OHEM) and pvanet. Furthermore, I used Xception in Keras to directly classify the whole images without any detection. Lastly, I trained a binary pure fish detector using Keras-Resnet 50.

When generating prediction, I rotated the images several times and averaged the outputs (same operation as data augmentation in the training). I added dropout layer to resnet-50/resnet-101 since the original protocol does not have any dropout. The final output probability is summation of py-RFCN, py-pvanet and Xception. Then I calibrate this output using the resnet-50 pure fish detector.

It is my first time to do vision problem. I have learnt a lot from this competition. Thanks for all the discussions and kernels. What makes me sad is that I currently rank 15 while the gold medal is awarded to first 14 (if my calculation is correct)....lol
