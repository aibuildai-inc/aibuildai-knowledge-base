# Thanks!

Competition: challenges-in-representation-learning-facial-expression-recognition-challenge
Rank: #3
Source: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/discussion/4676#25022

<p>Hi&nbsp;Yichuan,</p>
<p>Thanks for publishing source code and a link to the paper! It is a very nice idea - to train SVM and NN simultaneously. I looked into the sources, did I get it right that the structure of NN is the following?</p>
<ol>
<li>48x48x1fm </li><li>Mirror image </li><li>Rotation and scale (center &#43;-3 in both directions, angle &#43;-45 degree, scale 0.8-1.2), cropping result to 42x42x1fm
</li><li>Convolutional layer (fully connected wrt feature maps) with 5x5 weighting windows, 42x42x32fm, activation function - rectified linear
</li><li>Max subsampling layer, 21x21x32fm </li><li>Convolutional layer (fully connected wrt feature maps) with 4x4 weighting windows, 20x20x32fm, activation function - rectified linear
</li><li>Average subsampling layer, 10x10x32fm </li><li>Convolutional layer (fully connected wrt feature maps) with 5x5 weighting windows, 10x10x64fm, activation function - rectified linear
</li><li>Average subsampling layer, 5x5x64fm </li><li>Fully connected with 20% dropout of output neurons, 3072 output neurons, activation function - rectified linear
</li><li>Fully connected, 7 output neurons, activation function - L2 SVM </li></ol>
<p>I had rather similar structure of NN, the major changes are:</p>
<ol>
<li>I used hyperbolic tangent activation for the last layer </li><li>All other layers had &quot;hyperbolic tangent &#43; rectification (absolute value)&quot; activation applied (it seems it is time for me to stop being lazy and implement rectified linear).
</li><li>The first fully connected layer in my NN was _much_ smaller than yours: just 128 neurons. That's a huge difference.
</li></ol>
<p>May I ask, did you introduce this 3072-neuron layer to have LinearSVM working fine? Or you had this layer even when softmax activation was applied to the last layer? I would expect serious overfitting in the second case&nbsp;even with dropout applied.</p>
<p></p>
<p>Thanks,</p>
<p>Max.</p>
<p></p>
