# Post Competition Architecture Discussion

Competition: passenger-screening-algorithm-challenge
Rank: #4
Source: https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/45805#261490

I decided to approach this competition from the viewpoint of object detection.  I annotated a subset of the training images by drawing bounding boxes around the threats in the aps images (I spent maybe 8 hrs on this).  I then trained Faster-RCNN using the annotated images, with the label of each object being its location on the body.  This way the network would learn to both detect objects and label them based on their location all in one go.

For each aps image there are 16 viewpoints.  I ran the object detector over each viewpoint in the aps image, to generate a set of candidate object detections, with each detection having a probability over each body location.  I took the maximum of all the detections for each body location to get a final 17 dimensional confidence vector.  I then took the 17d confidence vector for each viewpoint to get a 16x17 matrix.  I used this to train an 17 gradient boosted classifiers (one for each body location) to get better calibrated probabilities.

The only data augmentation I performed was doing a horizontal flip of the images while training the object detector.  This actually will change the label.  For example, if a threat is labeled "right ankle" then after flipping it will then be labeled "left ankle". 

I decided on using an object detector due to the limited size of the dataset and the small number of unique individuals.  Faster-RCNN is a two stage object detector.  In the first stage, it recognizes potential threats; the second stage labels the threats/decides if they are not true threats.  I thought that this built in attention mechanism would help to prevent overfitting by forcing the network to focus on the threats.

My final model consisted of an ensemble of 5 models.  Each model using 80% of the data to train the object detector and then the final 20% to train the boosted classifier.  I then averaged the predictions over each individual model.  However, ensembling did not really seem help the performance of my model in any significant way.

Now some things that I tried that didn't work.  I spent some time trying to use the 3d data.  I used a 3d inflated convolutional network based on the vgg16 architecture.  This just means that I turned each 2d convolution into a 3d convolution and initialized the weights of each 3d convolution by stacking the weights of the pretrained vgg model.  I then tried to use this network to perform 3d object detection.   This best I was able to do was around 0.06 on my validation set as opposed to 0.025 using my 2d model.
