# My  segment and count solution

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #3
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35448

Congratulations to outrunner,konstantin,thanks to threeplusone for the lion coordinates,thanks to kaggle and noaa for hosted a very good competition.<br>

Here is my brief overview of my solution.<br>

**My goal of this competition:**<br>
I want to try my best to find and count all sealions .<br>

**Challenges:**<br>
1.Scale variance of the test data which lead to no stable local cross-validate method.<br>
2.Pups hard to find.<br>
3.In some images sealions are close to each other,and those images have large weights to RMSE.<br>
4.Sealion types almost can only be distinguished by their size,especially adult-females and juveniles.<br>

**Result:**<br>
1.My model can find almost all sealions and get their outlines(masks) except pups.<br>
2.I did not find a way to handle scale variance.<br>

Best private/public scores:13.03/14.03<br>

**Some demos:**<br>
Some dotted test set images:<br>
![3190 color dotted][1]
![361 color dotted][2]
The masks,outlines predicted by my models:<br>
![outline demo 1  ][3]
![outline demo 2 ][4]
The sealion-outline-star relation<br>
![relation ][5]
**The way to find all sealions:**<br>

My final solution base on two UNet:<br>
The first one, to predict dot on sealions,I called it star-unet<br>
The second one:to predict the the mask of each type of the sealions(I labeled all training set sealions mask) I called it outline-unet<br>
<br>
The two net’s structure are almost the same,the difference is the first net is  logistic regression and the second is 6-way softmax regression.<br>
 
As to star-unet,I generated the mask by setting the value of the mask to 255 at the center of the sealions,and then  decreasing to 255*0.8,255*0.6,255*0.4 as the position go far away from the center.<br>

As to outline-unet:I labeled the sealion’s outline step by step.I labeled 3 images and trained the net on them first,and let the net predict the mask of the some train-set images,and I modified the predicted images’ mask,when I have labeled 100 images,the result were very good.<br>


**The way to count the sealions:**<br>
Because my segment net is very good,so I decided to count the sealions based on them,I used a lot of functions of skimage(especially morphology) and scipy.ndimage. <br>
It’s not a easy job to split the sealions outline when they are connected,I used the star predicted by star-unet and used a lot of skills on morphology’s closing/opening and scipy.ndimage.label,and finally found more than 1 million sealions. I guess the recall-rate &gt;80% except pups.<br>
<br>
As to pups,I predicted the mask with different scales and ensembled them.<br>
<br>
**The pities:**<br>
I failed to find a good way to get the scale factor of every test image.I planed to find the sealions with large area in every test image and compared to the average areas of the adult-males and get the correct scale of every test image(or similar ways)BUT I had no time to finish it,because I under-estimated the difficulty of this competition.I got 15.74 public LB  using two Faster-RCNN network and then I entered another competition, when I was back, there were only 3 weeks left and the test-set is a big one.<br>
<br>
So,my final submission is simple ensemble of the prediction with different scale factors.It’s not much better than single model.My best single model is 13.21/14.26. <br>
<br>
**Other methods I tried**:<br>
I trained two Faster-RCNN networks with Resnet-101,but I found it can not handle sealions that stay close, I can not count them and  can only estimate them,I thought I should find better way,so they are not appeared in final ensemble.


  [1]: https://raw.githubusercontent.com/bestfitting/kaggle/master/sealions/3190.png
  [2]: https://raw.githubusercontent.com/bestfitting/kaggle/master/sealions/361-580.png
  [3]: https://raw.githubusercontent.com/bestfitting/kaggle/master/sealions/48-mask-outline.png
  [4]: https://raw.githubusercontent.com/bestfitting/kaggle/master/sealions/6983-mask-outline.png
  [5]:https://raw.githubusercontent.com/bestfitting/kaggle/master/sealions/star-demo.png
