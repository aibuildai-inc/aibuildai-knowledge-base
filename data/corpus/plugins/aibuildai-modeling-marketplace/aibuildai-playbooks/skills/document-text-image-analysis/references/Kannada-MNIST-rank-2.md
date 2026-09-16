# 2nd place Solution

Competition: Kannada-MNIST
Rank: #2
Source: https://www.kaggle.com/c/Kannada-MNIST/discussion/122230

Many thanks to the organizers of the competition. Thanks to Vinay Uday Prabhu for providing dataset and an interesting article on arxive.org.


I used CNN architecture (+RMSProp,+ReduceLROnPlateau) from kernel of FWiktor (thanks a lot to him).  This kernel has now ceased to be public. Here is the structure of this network:



Data augmentation was performed with these parameters:
```
ImageDataGenerator (rotation_range = 10,
width_shift_range = 0.25,
height_shift_range = 0.25,
shear_range = 0.1,
zoom_range = 0.25,
horizontal_flip = False)
```

At first, three sets of weights w1, w2, w3 were generated:

**1.**  
- 45 epochs, test_size = 0.05 -&gt; w1
- 60 epochs, test_size = 0.001 -&gt; w2
- 80 epochs, test_size = 0.0005 -&gt; w3

Next, 2 more sets  of weights  were generated (w-pseudo, w-pseudo-corr ):

**2a.**  
- ensemble 3 models w1+w2+w3
- pseudo labelling (thanks a lot to Nandor Balogh)   with threshold=0.95*3
- 100 epochs, test size = 0.0005 -&gt; w-pseudo


**2b.**
- ensemble  3 models w1+w2+w3 
- correction of posterior probabilities:
```
results [:, 4] = results [:, 4] - 0.7*3
results [:, 9] = results [:, 9] - 0.7*3
results [:, 1] = results [:, 1] - 0.7*3
results [:, 2] = results [:, 2] - 0.7*3
```
- pseudo labelling with threshold=0.95*3
- 100 epochs, test size = 0.0005 -&gt; w-pseudo-corr

And at the end, calculation of two submissions:
**3a.**  without correction
 - ensemble 2 models: w3 + 3*w-pseudo -&gt; submission  -&gt;  0.9918/0.9928

**3b.** with correction
- ensemble 2 models: w-pseudo + w-pseudo_corr
- correction of posterior probabilities:
 ```
 results[:,1] = results[:,1]*0.3
 results[:,2] = results[:,2]*0.2  
 results[:,9] = results[:,9]*0.05 
```
- submission - &gt; 0.9936/0.9952



Later, I found on the Internet a number of articles where correction of posterior probabilities is applied. Here is one of them: A Posteriori Corrections to Classification Methods https://www.researchgate.net/publication/250423867_A_Posteriori_Corrections_to_Classification_Methods
