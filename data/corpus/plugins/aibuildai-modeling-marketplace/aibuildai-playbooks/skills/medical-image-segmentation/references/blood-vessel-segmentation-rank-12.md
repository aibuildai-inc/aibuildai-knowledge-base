# 12st Place Solution for the  SenNet + HOA - Hacking the Human Vasculature in 3D

Competition: blood-vessel-segmentation
Rank: #12
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/476457

# Context

This solution was implemented as part of a blood vessel segmentation competition organized by the Common Fund’s Cellular Senescence Network (SenNet) Programm in cooperation with the Human Organ Atlas (HOA). 
Competition overview page: [SenNet + HOA - Hacking the Human Vasculature in 3D](https://www.kaggle.com/competitions/blood-vessel-segmentation)
Competition dataset is [here](https://www.kaggle.com/competitions/blood-vessel-segmentation/data)
Many thanks to the organizers for the opportunity!
# Overview
Framework — **TensorFlow**
Data pipeline — 2d, roi, resize (**1024x704**), **tfrecord**
Model — almost classic **U-net** (details below)
The solution is presented in two notebooks:
 - [train](https://www.kaggle.com/code/pib73nl/sennet-hoa-bvs-12th-place-solution-train)
 - [inference](https://www.kaggle.com/code/pib73nl/sennet-hoa-bvs-12th-place-solution-infer)
# Disclamer
This solution was developed in December 2023, before Santa's New Year's gift, which ultimately helped more than seven hundred participants to jump above 0.8. I left for the holidays in 238th place with a score of 0.567. When I next opened the leaderboard a week later, I had dropped over 150 positions! And that was just the beginning! ;) I worked on this approach for another week. I must say that I did this without much enthusiasm, since by improving the metric I found myself lower and lower in the ranking.
Finaly, with a result of 0.636, which was achieved by increasing the image size and minor architecture changes, I began to look for other approaches (see below in chapter *„Fruitless attempts“*).
# Data preparation
All data (except for the kidney_3_dense labels) were used as training data. The images have large fields that contain no useful information. To reduce these fields, the images were preprocessed to extract roi using statistical methods. 
```python
def apply_roi(image, label=None):
    """
    Exclusion of uninformative image fields 
    """
    # just throw out rows and columnt with low std
    row_mask = image.std(axis=1)>0.22
    clmn_mask = image.std(axis=0)>0.22
    
    # cleaning up the of noize of this approach and taking a solid region
    row_mask = cleaning_mask(row_mask)
    clmn_mask = cleaning_mask(clmn_mask)
    
    image = image[row_mask,:][:, clmn_mask]
    label = label[row_mask,:][:, clmn_mask] if isinstance(label, np.ndarray) else None 
    
    # remember the size of the pads for subsequent correct restoration
    row_pad = (row_mask.argmax(), row_mask[::-1].argmax())
    clmn_pad = (clmn_mask.argmax(), clmn_mask[::-1].argmax())
    
    return image, label, (row_pad, clmn_pad)

def cleaning_mask(mask):
    """
    Selecting a solid region from a noisy mask
    """
    # if frame starts from the first element or finishes at the last
    mask[0] = False
    mask[-1] = False
    
    # taking edges of frames
    frames = np.nonzero(mask[:-1]!=mask[1:])[0]
    # taking length of frames
    delta = frames[1:]-frames[:-1]
    # taking index of max len frame
    max_solid_block_begin = np.argmax(delta)
    # other is garbage
    garbage = np.delete(frames, [max_solid_block_begin, max_solid_block_begin+1])
    # clearing the mask
    for a, b in zip(garbage[::2], garbage[1::2]):
        mask[a+1:b+1] = False
    
    return mask
```
Next, all images were reduced to a single size of 1024x704. The experiments started with a size of 384x256, and as the size increased, the result expectedly improved. 1024x704 is the maximum size that did not result in an OOM error. An example of the processed image is below.
[Example of a processed image]
Every 25 images (4%) were used for validation, since the density of the labels varies greatly along the z-axis.
The resulting images and tags were packed into tfrecord files to organize a multi-threaded pipeline (total files - 92, 162 MB each). The maximum possible batch size for the 1024x704 shape turned out to be 32. Augmentation was not used - I just couldn’t get around to it!
# Model
The more or less classical **U-net** architecture was used as a model.
[U-net architect]
Losses were estimated using **binary crosentropy**. The **Adam** optimizer was used for optimization. The **learning rate** was changed according to the **cosine decay** schedule with warmup.
Since there is a significant class imbalance, weights were used. The idea was to set the weights at the instance level, since the class ratios vary greatly as we move from the center of the kidney to the edges (along the z-axis). But to begin with, I hardcode the weights, and it worked tolerably well. I didn't return to this issue later, so there is room for improvement.
The model was created from scratch and trained for 60 epochs. For prediction, epochs with a minimum value of validation losses were taken.
The prediction result looked something like this:
[Prediction result]
There are quite a lot of FP here… However, it makes sense to work on the sample weights 🤔
# Fruitless attempts
Obviously, given the large number of small details, any resizing harms the result. I tried to solve this problem by dividing the image into fragments (intersecting tiles of 256x256 size). I used the same model architecture. But the labels turned out to be exclusively in the places where the tiles overlapped, and having assembled a mask from the tiles, I got a blank sheet! I haven't had time to figure this out.
Second. I tried to solve the problem of label resizing by changing the architecture - I added another “kinda u-net” to the end of decoder - 2 convolution layers and two reconvolution ones. Didn't do well here either, but would have been in 67th place on the private leaderboard 😉

# Some observations
Yes, yes… There was a big quake… For some reason, most of the solutions failed in suspiciously similar ways ;) This is clearly noticeable in the interval of about 100-600 places.
[LB shake plot]
My main solutions, similar in network architecture and image size to the winning one, gave stable results on a public and private dataset. On a private dataset - even a little better!
[Score of the winning model]
The difference between the public and private data sets was 0.012 points. Such stable results in the first thousand can be counted on the fingers of one hand. In general, the variance is already normal, all that remains is to work on the bias 😁
Thanks to everyone who worked on the problem! It was interesting with you! Good luck! ✋
