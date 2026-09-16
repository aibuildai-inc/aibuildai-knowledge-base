# No Conv Net (Private: 4th,  Public: 2nd)

Competition: trends-assessment-prediction
Rank: #4
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162836

## Some thoughts
We will try to make this dicussion post as comprehensive as possible and answer every question in the comment section. We had to shorten some parts as this would get way to long. 

It is very interesting that 3d CNNs seem to perform better on Private LB compared to our approach. We thought they would overfit more. It seems like Private LB distribution is closer to Train distribution than to Public LB distribution. 

## Credits
First of all thank you to the organizers for this great competition and hughe shoutouts to @churkinnikita  and @simakov for their well deserved win. It was a very memorable moment for us when they overtook our Public 1st place. 

Thanks to @aerdem4  for the amazing kernels and Kudos to everyone who participated in this competition.

## About us
Marius and I are lab buddies. We wrote our medical doctor thesis together in the neurophysiology department. That's why we have worked with neuroimaging data before (Fortunately this is not Zillow)

## Approach

We wanted to create a comprehensive method for predicting fmri images to be able to directly evaluate, which parts contributed the most to the models prediction and not finetune for Site1 and Site2 but build 1 model for both of them.

In the brain it's all about connectivity. Normally fmri images are 4 dimensional with 1 axis being time. But here these were ICA components - different brain networks. Marius had the idea to consider this axis like a time axis and use the same tools to derive features from time used in neuroimaging but for the ICA components. The keyword is Parcellations. Parcellations are an atlas of the brain, they subclassify the brain in smaller networks using functionality and/or anatomical features. Is is then possible to calculate the mean of fmri data points for every component in the atlas. Thereby we reduce the shape of  53*63*52*53  to Number of Parcellation components (a few hundred depending on the atlas) x Number of ICA components. In the following we will refer to this array as PICA. This reduces inter subject and Site variation.

We used several atlases to diversify our model to better generalise (we used up to 10 different):

### Basc:


### Schaefer:



We even calculated our own Parcellations using unsupervised learning approaches such as KMeans and Ward. We will link you some papers here as describing this would be beyond the scope of this post: 

Bertrand Thirion, Gael Varoquaux, Elvis Dohmatob, Jean-Baptiste Poline. Which fMRI clustering gives good brain parcellations ? Frontiers in Neuroscience, 2014.

Vincent Michel, Alexandre Gramfort, Gael Varoquaux, Evelyn Eger, Christine Keribin, Bertrand Thirion. A supervised clustering approach for fMRI-based inference of brain states.. Pattern Recognition, Elsevier, 2011.

### Kmeans



### Ward


We have seen this idea of atlases emerge in the discussion. But participants did not know what to further do with them. We had several approaches on this. We will only list the most contributing here, as this would be beyond the scope of this post: 

### Calculate connectivity of different components
Normally connectivity is derived by looking at the interaction of 2 components derived from parcellation over time. Here we used the ICA components as our “time” feature. We looked at the interaction of 2 components over ICA components using Matrices. 

We will visualise how this looks like using this basic yeo Parcellations with only 17 components. We used matrices up to 444 components as our RAM could not handle more.

### Yeo 



### 1. Correlation Matrix
The x and y axis are the different yeo components and the matrix is the correlation of these 2 components over the 53 ICA Networks.




### 2. Covariance Matrix




From now on only links as this gets too long :D

### 3. Partial Correlation Matrix

https://en.wikipedia.org/wiki/Partial_correlation

### 4. Tangent Matrix

G. Varoquaux et al. “Detection of brain functional-connectivity difference in post-stroke patients using group-level covariance modeling, MICCAI 2010.

### 5. Precision Matrix

https://en.wikipedia.org/wiki/Precision_(statistics)

… you get the idea

### What to do with these Matrices

We then used these matrices as input for our model (SVR, Ridge, Lasso, Elastic Net) and stacked their predictions using SVR and Ridge with a simple blend in the end. 


We also fed the raw parcellation data to some networks.

And we used the public notebooks for IC and FNC features


## Results
with Lofo: CV: age: 0.13378; d1v1: 0.14976; d1v2: 0.15016; d2v1: 0.1795; d2v2 0.175050 overall: 0.154682; Private LB: 0.15714; Public LB: 0.15615

without Lofo:  CV: age: 0.13477; d1v1: 0.15027; d1v2: 0.15077;  d2v1: 0.18045; d2v2 0.17545 overall: 0.15539; Private LB: 0.15689; Public LB: 0.15626


## Some more Tricks
1. Leave one Network Out
2. Only use a single Network
3. Calculate our Parcellations using Site 2 
4. ...

## What we think contributed the most to not overfit to Site1 / Site2: 

1. using Brain Atlases derived from Site1 + Site2
2. using Brain Atlases derived from completely different experiments (e.g. Basc)
3. The shear amount of features used in our models
4. The amount of base models (around 70)
5. The amount of parcellations used

## What did not work

1. Calculating our own Parcellations solely on Site 1 -&gt; overfit to Site 1. 
2. We tried a lot with 3d CNNs but were not able to improve CV using them. - We were especially interested in remapping the Parcellated brain back to 3d. But this did not give any improvement. We are excited to see how other dealt with this. 
3. Autoencoders in many combinations
4. ....


## Discussion: 
We have not seen this approach anywhere in the literature. We now can perfectly calculate which components were contributing to the models prediction. Which brain regions are important and what interaction between different brain regions are relevant for ageing and the other assessment values. We hope that the organizers will allow us to further develop this idea and discuss the results with our medical supervisors. 


# Cheers!




PS. One of the most memorable moments of this competition for us will be the Japanese dominance in the LB
