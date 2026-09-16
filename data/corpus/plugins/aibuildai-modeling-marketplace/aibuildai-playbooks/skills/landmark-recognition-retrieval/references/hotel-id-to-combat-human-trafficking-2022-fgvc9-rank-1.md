# 1st place solution - New augmentation method + 5 model ensemble

Competition: hotel-id-to-combat-human-trafficking-2022-fgvc9
Rank: #1
Source: https://www.kaggle.com/c/hotel-id-to-combat-human-trafficking-2022-fgvc9/discussion/328281

Thanks to the sponsors and Kaggle for organizing this important research challenge.

***TLDR***: Solution architecture includes 5 model ensemble at two different image sizes with embedding dimension reduced by PCA.  Secret sauce is a newly developed augmentation method designed to deal with large occlusions while maintaining semantic cohesion of the overall scene.

## Augmentation - Introducing BlendFlip:
The core problem in this challenge is dealing with large masked occlusions in test/query images.  For this my goal was to inpaint the masked region while attempting to:
- maintain semantic cohesion with the overall scene
- maintain continuity around masked edges
- fill in visually realistic content

When looking at hotel room images, a common pattern is many vertical and horizontal features against repeating backgrounds.  For instance wall corners that run vertical, door frames that run vertical and horizontal, pictures on walls that are square, etc.  This observation gave me the idea to apply image patches from the same image directly adjacent to the masked region and essentially flip them into the masked region, and then blend the flips from the vertical and horizontal direction, hence the name BlendFlip.

[BlendFlip]

BlendFlip algorithm:
1. Determine max distance between masked region edges in the vertical direction (image top to mask top, mask bottom to image bottom).  Do same for horizontal.  Max direction becomes the regions to be used for inpainting.
2. Flip max vertical region into masked region.  If initial flip doesn’t fill entire masked region, keep flipping in vertical region until masked region is filled, cut off at edge of masked region.
3. Repeat step 2 for horizontal region.
4. Take 50/50 blend of step2 and step3 (cv2.addWeighted) as final inpainted region.

BlendFlip helps address 2 1/2 of the 3 problems listed above.  It helps maintain semantic cohesion with visually realistic content.  It maintains continuity on 2 of the 4 sides of the masked edges.  I tried GAN's, traditional CV inpainting methods, variations of CutMix, but BlendFlip significantly outperformed them all.

[Example 1]
[Example 2]
[Example 3]

The goal of blendflip is NOT to predict what might be in the occluded region, but rather maintain the integrity of the image embedding from the non-occluded regions.

## Solution architecture
The overall solution architecture is shown in the diagram below.  Images of sizes 1024x1024 (longest max side scaled to 1024, aspect ratio maintained) or 384x384 are fed into 5 models of various architecture.  Each model is trained with ArcFace loss and the embedding size of each is 1536D.  The embeddings are then concatenated and reduced to a total of 3072D via PCA (size determined by maintaining 99% of variance of embedding concat), and then KNN performed.  Each model is trained with a 50% probability of BlendFlip on each image.  BlendFlip is performed on all test images prior to inference.  No post processing or re-ranking was applied. 

[Solution architecture]

For the CVPR paper I will include proper ablation studies and relative impact of the solution ingredients.  BlendFlip accounted for ~0.03-0.04 mAP.
