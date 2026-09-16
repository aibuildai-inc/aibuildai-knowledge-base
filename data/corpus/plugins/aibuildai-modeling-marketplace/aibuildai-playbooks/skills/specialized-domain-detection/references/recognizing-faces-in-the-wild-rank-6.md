# #6 solution review | quick notes | private 0.917

Competition: recognizing-faces-in-the-wild
Rank: #6
Source: https://www.kaggle.com/c/recognizing-faces-in-the-wild/discussion/103457#latest-598137

First of all thanks @hsinwenchang for sharing the kernel - that's what I started off with. Thanks to @CVxTz too who originally shared the code in github.

### First, things that didn't work :

**1. GlobalMinPool** - Using Avg and Max pool helped a lot. And I thought of adding a Min pool to extract a bit more relevant information - but, this worsened the performance - by a lot. I don't really have a reasoning behind this, but, I would assume that since the resnet structure doesn't use min pool - this wasn't able to really help here.

**2. SeNet** - wasn't able to make the senet50 structure work in keras_vggface - the accuracy would always stay around ~50%, although the AUC would sometimes rise to 59-60%. I think that's a documented issue in rcmalli's package itself. 
But, what was surprising was that even porting pretrained pytorch models had similar problems. I think because all these models were ported from original Caffe models which expects different ranges of tensor than what Pytorch expects - might be the issue there. Then again, it's just a hunch. 

***If anyone was able to successfully train senet (in keras or pytorch), do let me know what you got right.***

**3. Nadam/SGD/LAMB** - tried out multiple oprimizers, but ultimately Adam with 1e-5 starting lr gave me the best performance.

**4. Focal Loss** - Improved my public LB from 0.917 to 0.919 - but, ultimately in private gave (a slightly worse) 0.917


### What worked:

**1. distance metrics** - Using x1.x2, (x1-x2)^2 and x1^2-x2^2 in combination worked out the best for me. I think adding x1*x2 gave almost 0.1 public LB boost for me.

**2. Progressive freezing** - Contrary to what I have seen in transfer learning, progressive freezing of layers gave me better performance than progressive unfreezing. It was surprising indeed, but, I think that makes sense since this was not a 'pure' transfer learning exercise - rather the base model had to be repurposed to gauge the differences between faces.

**3. Data augmentation** - Flipping, Blurring and Rotating worked out the best for me. Given we had BW images in the dataset, I tried randomly greyscaling the images, but, that didn't work out at all. 

**4. Image size** - contrary to what I read in the discussions, image size of 224 gave the best results for me.
