# 6th solution, objection detection + cnn classification

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #6
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35545

Thanks to @threeplusone for your coordinates.

My solution consists of two parts, firstly using faster rcnn to get candidate box, and then using cnns to classify.

Two faster rcnns(AN and BN) are trained. AN is trained through multi – scale input ,zooming in each picture two to four times randomly. BN is trained through single – scale input,zooming in each picture twice. Because The scale of the same age group of sealions in different picture changes greatly, for example, in Test ,the adult’s size vary from ~100 pixels to ~300 piexls. When sealions are too big, my single-scale network does not completely cover the whole sealion (mainly adult male),but it has less false positives. Because in second stage I need to use box’s size, I train the muti-scale faster rcnn . It completely covers the sealions, but it has more false positives.

And then I train a CNN to reduce false positives.

Finally, I train about a dozen cnns to classify candidate boxes from AN. I choose best four them lastly and average their results.

Four cnns are as follows,

1)72x72 input, vgg-like cnn

2)72x72 and 144x144 muti - input network

3)72x72 and 72x72(108x108 patch scaled to 72x72) muti - input network

4)same as net3, but output five classes(no background)

In test stage, I will scale each picture at a certain scale(scale = target-scale/ max_box_length , max_box_length is the longest edge of all the candidates from BN in a picture, target-scale is about 78-88). The candidate boxes' coordinates are scaled at the same scale.
