# 2nd place entry (code+paper)

Competition: challenges-in-representation-learning-the-black-box-learning-challenge
Rank: #2
Source: https://www.kaggle.com/c/challenges-in-representation-learning-the-black-box-learning-challenge/discussion/4726

<p>I attach my submission code and workshop paper to submit. Because the paper is a draft version, please do not link this version of paper on blog or website. And though I used this submission code really, the code used in the paper is a little bit different from this. I will post the well-organized code with the final version of my paper&nbsp;later.</p>
<hr>
<p><strong>The full description of my approach</strong></p>
<p>At first, I used 1 hidden layer neural network with rectified hidden unit + softmax output unit. The private score of pure supervised learning with 1000 labeled data is 0.53~0.57. At that time I used simple mini-batch SGD with fixed learning rate + momentum.</p>
<p>For unlabeled data, I was searching for a simpler way of semi-supervised learning without pre-training. So I tried <strong>pseudo-label</strong> method and ranked 1st at that time. For the performance, I used <strong>sigmoid output unit</strong> (inspired by CAE) and <strong>simultaneous training with labeled and unlabeled data.</strong></p>
<p>The important thing is that <strong>pseudo-labels are re-calculated every weights update</strong>. If we calculate pseudo-label once after training with only labeled data, pseudo-label might be less accurate because the network is overfitted. After training several initial epochs with only labeled data, the network should be trained with labeled data and unlabeled data using continuously re-calculated pseudo-label. This scheme improve the generalization performance really. Private score is ~0.65</p>
<p>In training with pseudo-label, <strong>the balance between labeled data and unlabeled data is also important.</strong> Because the number of unlabeled data is far more than labeled data, mini-batch size also should be larger. And balancing coefficient in loss function should increase linearly according to epoch. (detail of this scheme is in the paper.)</p>
<p>Next key ingredient is dropout. <strong>dropout</strong> is an amazing technique for supervised learning of deep neural network. Superficially my method looks like supervised learning, dropout technique can boost up my method significantly. Private score is ~0.6844</p>
<p>To enter winning entries, I used also <strong>polarity splitting</strong>. (not included in the paper )&nbsp;Some nice papers for image recognition use polarity splitting. (inspired by &quot;The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization, Adam Coates and Andrew Y. Ng. In ICML 28, 2011.&quot;) In the case of rectifier unit, minus part of net activation is not used. Then additional features using -W are almost always helpful. For this, I trained the network one more with W and -W. So the number of features was doubled. Private score with this technique is ~0.6958.</p>
<hr>
<p>&nbsp;</p>
<p><strong>* My code requires jacket - GPU matlab toolbox</strong>. It is easy to convert to CPU version : grand -&gt; rand, gsingle -&gt; single, grandn -&gt; randn, gzeros -&gt; zeros.&nbsp;</p>
