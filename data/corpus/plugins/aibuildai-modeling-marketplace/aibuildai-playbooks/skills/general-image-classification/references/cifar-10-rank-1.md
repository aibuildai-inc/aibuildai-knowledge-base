# Train you very own deep convolutional network

Competition: cifar-10
Rank: #1
Source: https://www.kaggle.com/c/cifar-10/discussion/10493

<p>Attached is a (new) implementation of a &quot;spatially-sparse&quot; convolutional network (http://arxiv.org/abs/1409.6070).</p>
<p>Step 1) Look at the README file.<br>Step 2) Define a network architecture in terms of LeNet style layers. The input layer can be much larger than 32x32, and it will still run relatively efficiently (sparsity!). Be sure to include lots of Network-In-Network layers. [See runCifar10Kaggle.cu]<br>Step 3) Specify some kind of affine distortion to apply to the training data. [See OpenCVPicture_AffineTransform.h]<br>Step 4) Hope it all compiles. <br>Step 5) Run it for a long time; hope it does not crash.</p>
<p>Requirements: CUDA (sm_20), OpenCV, Boost and a reasonably fast GPU</p>
