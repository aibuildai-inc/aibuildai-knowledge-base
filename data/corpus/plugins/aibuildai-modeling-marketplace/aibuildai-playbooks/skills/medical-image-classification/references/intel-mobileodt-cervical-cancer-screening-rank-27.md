# 27th place solution: Polar Coordinates

Competition: intel-mobileodt-cervical-cancer-screening
Rank: #27
Source: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/35269

Actually placed at 26th after LB update. 
Idea was to exploit the approximate radial symmetry in the problem to a number of advantages that I would describe below, of course, with some definite LB gain! (Any other physicists in the room!?) Polar coordinates seem to be a "natural" frame for the features that were important to the cervix type detection (transformation zone being a ring).

I think there is a lot more scope for improvement here as I very hastily implemented this in the last few days of the competition. Posting this in case it turns out helpful for MobileODT in some way.

## First, a few Basic transformations applied on the Images: ##
$ Since images could be green filtered images, and it seemed color didn't add any information, I had replicated the green channel for R, G, B (so that I can make use of the pretrained models), instead of using grayscale values for R, G, B as greyscale values of a green filtered test images would be from a different distribution then, as average = (g + 0 + 0)/3 in that case, making the mean 3 times smaller than expected distribution. 

$ These G, G, G channel images were then resized to a maximum dimension of 598x598 with aspect ratio preserved.

$ Then a center crop of 448x448 (75% of 598) was extracted without any resizing.
These were used as the base images before any more processing.

Then trying out a bunch of pretrained finetuned models (Vgg16BN, InceptionV3, ResNet, Xception) on these base "cartesian" images gave a best single model of around 0.69-0.71 PBLB at best, which gave a fairly good score after weighted ensembling. For tuning these weights of the ensemble, I had used a separate hold-out set of data. At this point adding any other model to the ensemble didn't add much to the final score.

## Polar Coordinates ##
A polar transform is a map (not one to one, actually) that takes (x,y) from Cartesian coordinates to (r, theta) in polar coordinates as follows:

$$ (x,y) \to (r,\theta)=\left(\sqrt{x^2 + y^2}, ~\text{tan}^{-1}\frac{y}{x} \right) $$.

Notice that this is not a one to one map (it is a one-to-many map). For example, $$(r, \theta), ~\text{and} ~(r,\theta + 2n\pi)$$ trivially maps to the same point in cartesian coordinates. Also, the center/origin in x-y coordinates maps to an infinite number of points  $$(r=0,~\theta)$$ for any theta (basically the theta axis).

At first I tried to implement this in a python function (fairly straightforward), but then resorted to ImageJ's plugin [PolarTransformer.class][1] due to its fast pixel crunching.
An example of this transform on a spherically symmetric image:



One may analyze the effect of this on cervix images by considering the below illustrative picture where the blue/green represents some background noise (the arrows are drawn to indicate the direction of the convolutional kernels over the image):

The black region towards the right most end of the polar image is due to the fact that the original image is rectangular shaped and not a circular one, and hence there are always some missing pixels over the circular ring with radius r &gt; r_0; where r_0 = size/2 (size being the width or height of the square image in cartesian coordinates).

The noise can now be easily cropped and then checked for quality by reconstructing Cartesian from Polar using the inverse map which is a many-to-one map:
$$ (r, \theta) \to (x, y) = (r \cos\theta, r\sin\theta) $$
as follows:


The advantages with this transform are two-fold:

1 - Considering the green and blue parts above to be the black borders of the MobileODT tube or some non-cervix background, that can be now, easily eliminated by cropping out upto a limited value along the radius axis of the polar image (width) leading to less overfit models.
Example: 
Consider train_data/Type_2/1038.jpg:

(360 here corresponds to max theta in degrees)
In my case I used these 216x360 cropped images for bulding models over polar data. To inspect the quality of information in the cropped polar images, one may check by reconstructing images by inverse transform:

Pretty good reconstruction! The black background of the original image can now be seen painted in the reconstructed image, which is due to those pixel maps that have multiple cartesian coordinates for the same polar coordinate (many to one). 
One can easily find a good cutoff for such noise cropping manually as region around a cervix over a certain radius is anyhow not much important for cervix type classification (ofcourse assuming cervix centered images).

2 - Applying CNNs on polar images makes the square kernels "compatible" with the transformation zone ring. A kernel running over Cartesian images is represented by the green arrow above, while a kernel running over the Polar ones is represented by the yellow arrow.

But when pretrained models were built on these polar images alone, single best model was much worse than cartesian -  standing at around 0.78. But I was atleast sure this was not as much overfit.

To get the best features out of the cartesian and polar (216x360), finally, I resorted to using the flattened last convolution layers from two separately finetuned resnet models on cartesian and polar, and feeding them to xgboost. 
Adding this model to the ensemble gave me about a 0.07 boost on the private LB.

The xgboost used was not cross-validated across various parameters as the competition was nearing end. There could be a scope for improvement here. 
Additionally, I didn't use any detect framework to get the cervix-centered crops, instead, I assumed images were more or less centered anyway and applied the polar transform. This is another possible area of improvement. Also, one may experiment with the reconstructed cartesian images from cropped polar because the pitch black background generally kills the kernels and overfits.

Let me know if you find these features helpful. Thanks!

  [1]: https://imagej.nih.gov/ij/plugins/polar-transformer.html
