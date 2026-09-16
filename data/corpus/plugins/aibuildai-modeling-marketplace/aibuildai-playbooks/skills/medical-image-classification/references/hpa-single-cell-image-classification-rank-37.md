# Private 0.458 with 3 single cell classifiers (37th place solution) - a data centric approach

Competition: hpa-single-cell-image-classification
Rank: #37
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238380

Thank you Kaggle and HPA for hosting this competition, and congrats to all the winners. 

I've taken this competition more seriously than any other competition in my Kaggle journey. This post will have a brief description of my approach, which is heavily data-centric. Although my standing isn't among the very best, I'm hoping this post will offer a unique perspective.

**Overview**

Since we are expected to predict labels per cell, I focused all my efforts entirely on building the best cell level image classifiers. Given the size of the HPA dataset, we can extract a lot of cell images. But, assigning the image level label to cell level image results in a lot of cell level images that are mislabeled. I initially trained models on images with a single label since they would have the least amount of noise. I was able to push my public score to 0.376 at most. 

However, it soon became clear that to increase my score even further, only images with single labels would not be enough. So, I started using images with 2 or 3 different labels. As mentioned before, assigning the image level label to cell level image results in a noisy dataset. And, re-labeling hundreds of thousands of cell level images by hand is tedious. So, I came up with various heuristics per class using which I could use to assess whether an image belongs to a particular class. These heuristics helped me select the best images from a lot of label combinations and make a less noisy dataset. Using this dataset, I trained 3 models: inception v3, mobilenet, and densenet121. The ensemble of these models gave private score 0.458 and public score 0.464.

**Selecting the best images using heuristics:**

The best part about this dataset is that they are all from the same domain. By that I mean that they are all images of cells. And, as a bonus, we have cell segmentation tools. From the images and HPACellSeg, we can get the following important locations of the cell: Cell mask (from segmentor), Nucleus (from segmentor), Cytosol (Cell mask XOR Nucleus mask), Microtubules (red) and Endoplasmic reticulum (yellow). By looking at the intensities of the green channel in these areas, we can judge whether an image belongs to a specific class with a high confidence. The following are some of my best heuristics:

**Nucleoplasm:** High green average in nucleus, whose location can be found using segmentor.
**Nuclear membrane:** High green average in the border of the nucleus mask.
**Nucleoli and Nuclear Fibrillar Center:** High green average in the nucleolus, which usually occurs as a dark spot inside the nucleus.
**Microtubules and Endoplasmic reticulum:** High similarity between green channel and red/ yellow channel. Using a hasing algorithm like PHash works well.
**Cytosol:** High green average in the cytosol (Cell mask XOR Nucleus mask).
**Mitochondria:** High green intensities in the microtubules.

These heuristics are no way perfect, but they do help in removing incorrectly labeled images and outliers. As an example, consider the label '2|16'. Good quality cell level images of this label should have a high green staining in the nucleoli and cytosol. Therefore, we can discard images that don't have either of those. To give a more specific example, [this](https://drive.google.com/file/d/14Hs2mnJz2KeoW6rUU1LIln5f3aYGQLGO/view?usp=sharing) image, has a higher green staining in both these locations compared to [this](https://drive.google.com/file/d/15xMAkD9xXGbEoGa6JVcTbs7vkgc9EW_c/view?usp=sharing) one. So, we can keep images like the first and discard those that are similar to the second.

I wasn't able to come up with amazing heuristics for all classes. For rarer classes like Mitotic spindle, I had to rely on hand labeling. I ended up with a dataset of ~250k cell level images consisting of around 144 unique label combinations. I used this to train an inception v3, mobilenet, and densenet121. The images were resized to 448 x 448 and simple augmentations (rotations, flips) were used. The ensemble of these models gave private score 0.458 and public score 0.464. 

**Conclusion**

Due to time and resource constraints, I wasn't able to experiment with too many other models. Combining image level and cell level predictions seemed to be an approach used by almost all, but I got the above score without image level models. My cell level model submissions only improved very little (and one time decreased). Clearly, I did not have too much success with combining image level predictions. But, it seems like those who were able to use image level and cell level images got the best scores. 

You may say that this is an extreme focus on the data, and it is. But, I felt it was necessary for me if I wanted a good score as I do not have access to expensive hardware (I had to rely on my HP laptop, kaggle GPUs and Colab).  Filtering out images improves the quality of the dataset while using a lower quantity of images without sacrificing model performance. 

Compared to other top solutions, my models are simpler (no transformers, no segmentation models, no multiple losses, etc). If you've had success with combining image level and cell level predictions without focusing on data preprocessing, I encourage you to try training on [my dataset](https://www.kaggle.com/novice03/clean-data). I cannot guarantee that all the images are correctly labelled, but I ensure you that there are a high proportion of correctly labelled images and also images from rare classes from the external dataset.

Finally, I'd like to thank @thedrcat and @dschettler8845 for their notebooks and @its7171 and @rdizzl3 for their datasets and the competition hosts for helping all of us.
