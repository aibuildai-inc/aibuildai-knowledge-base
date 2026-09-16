# So, what were your approaches?

Competition: galaxy-zoo-the-galaxy-challenge
Rank: #28
Source: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/discussion/7599#41550

<p>I trained a convolutional neural network with the following architecture:</p>
<p>8C9-S2-16C5-S2-32C5-S2-64C4-F37</p>
<p>where 8C9 means a convolutional layer with 8 kernels of size 9x9 pixels. S2 means a sub-sampling layer with a scaling factor of 2 (I used average pooling). F37 means a fully-connected layer with 37 output neurons. I used the MATLAB implementation by Rasmus given here (https://github.com/rasmusbergpalm/DeepLearnToolbox).</p>
<p>I preprocessed the images slightly (to get to 64x64 pixels) before inputting them into the network, used three input images (one for each color), and I simply modelled the problem as a regression problem with 37 outputs.</p>
<p>For training I used 55,000 images each rotated 4 times (giving 220,000 images in total). When predicting on the test set I inputted each image with 4 rotations and made a flat average. I finally post-processed the predictions to match the 11 constraints specified by the decision tree.</p>
<p>The biggest issue was training time, which reached 2 weeks on a university cluster. The implementation we used ran only on CPU and was written in MATLAB. I used quite a bit of time trying to optimize a more clever cost-function by using 11 softmax units and then deriving correct gradients, but this did not give anything useful. I also tried ensembling together multiple network architectures - this did not help significantly either.</p>
<p>Edit: I gave a more thorough description of my approach here:&nbsp;<a href="http://www.davidwind.dk/?p=49">http://www.davidwind.dk/?p=49</a></p>
