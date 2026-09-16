# Contest methods

Competition: SemiSupervisedFeatureLearning
Rank: #1
Source: https://www.kaggle.com/c/SemiSupervisedFeatureLearning/discussion/959

<p>Surprisingly, one supervised method we tried on a whim to establish a supervised baseline ended up as the top performing model. &nbsp;Here's a brief description of our method:</p>
<div>Though we explored various unsupervised and semi-supervised options, our best submission consisted of purely supervised features: the posterior probabilities output from Breiman's Random Forest algorithm. &nbsp;Input features to this algorithm came from the
 union of two sets: the top k features with the most non-zeros, and the top k features with the largest difference between class means. &nbsp;The sum of all features for each data point was included as well. &nbsp;Only the labeled dataset was used to select these features
 and train the model.</div>
<div>A total of 5 submissions were included as final features, each comprising a different number of the top features from the two feature selection methods (ranging from ~600 to ~1400 total features) and&nbsp;slightly&nbsp;different random forest parameters.</div>
<div>This method got first on the public and private sets, with a private AUC of 0.9771.</div>
<div>Here's several of the alternatives we explored:</div>
<div>
<ul>
<li><span style="direction:ltr">Weighted k-means / mini-batch k-means</span> </li><li><span style="direction:ltr">Wrapper methods around supervised methods to incorporate unsupervised data</span>
</li><li><span style="direction:ltr">Wrapper methods &#43; multiple views around supervised methods</span>
</li><li><span style="direction:ltr">SVD</span> </li></ul>
</div>
<div>Here's some of the other methods that we did not have the time and/or computational resources to explore, but we wanted to and are curious to see if other contestants gave them a shot:</div>
<div>
<ul>
<li><span style="direction:ltr">Sparse autoencoders</span> </li><li><span style="direction:ltr">Deep belief networks</span> </li><li><span style="direction:ltr">Restricted boltzmann machines</span> </li><li><span style="direction:ltr">Latent dirichlet allocation</span> </li><li><span style="direction:ltr">Self-organizing maps</span> </li><li><span style="direction:ltr">Graphical models</span> </li></ul>
</div>
