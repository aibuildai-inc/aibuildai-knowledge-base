# 13th place solution: 4-panel solo model

Competition: blood-vessel-segmentation
Rank: #13
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475117

First I would like to thank the host for this wonderful competition, I really enjoyed participating in it! I would also like to thank @hengck23 for his valuable and interesting comments, I learned a lot from this! As my solution was inspired by the [winning solution](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430618) of the contrails competition 6 months ago, I want to thank @junkoda as well! 

**4-panel image**


Inspired by @junkoda winning solution, the model was trained on 4-panel images consisting of 256px sized patches of consecutive slices (creating 512px by 512px images). Therefore, I was able to keep 2.5D dimensionality in a 2D image. The idea was that the model would learn this relationship between slices and therefore would predict a more continuous segmentation.
The 256px patches were made with the [EMPatches](https://github.com/Mr-TalhaIlyas/EMPatches) library. Which made stitching the separate 256px patches from each 4-panel image a lot easier.
The images were normalized based on the percentile of the whole kidney. All values lower than 0.5 were clamped to 0.5, to normalize the background better:
```
lo, hi = np.percentile(kidney_volume.numpy(), (2, 98))
def preprocess_image(image, lo, hi):
    image = image.to(torch.float32)
    image = (image - lo) / (hi - lo)
    image = torch.clamp(image, min=0.5)
    return image
```

**Augmentations**
I used simple augmentations for training: 
```
train_transform = A.Compose([
    A.RandomRotate90(p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightness(p=1),
    A.OneOf(
        [
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),
])
```
What did not work for me were the augmentations based on scaling of the image.

**Submission**
For the submission part: each patch in the 4-panel image was rotated on each own. Where after, the mean was taken of the rotated patches. The mean was taken as well of all the patches in the separate 4-panel images. These patches were merged with the ‘max’ setting. This was performed for the xy, xz, yz rotations of the whole kidney volume.

**Model**
The model that was trained on these 4-panel images was a Unet maxvit_tiny_tf_512 using segmentation models pytorch (SMP). The model was trained on 3 whole kidney volume rotations with 0.4 overlap in the patches (~490.000 different images). The model was trained for 9 epochs with a 1e-4 lr and then another 6 epochs with CosineAnnealingLR to 1e-6.

* Link to Kaggle submission notebook: [SenNet-HOA23 | 2.5D 4-panel | submission](https://www.kaggle.com/code/menno1111/sennet-hoa23-2-5d-4-panel-submission/)
* Link to the training and validation notebooks: [GitHub](https://github.com/Menno-Meijer/SenNet_VasculatureSegmentation_Competition)
