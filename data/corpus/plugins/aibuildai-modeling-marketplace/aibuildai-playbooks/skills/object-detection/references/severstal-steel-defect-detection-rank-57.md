# Silver frankenstein's monster (57th place)

Competition: severstal-steel-defect-detection
Rank: #57
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114517

This solution is actually a small modification of great kernel by @lightforever (https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672) which used mlcomp/catalyst 🤘.


Things i added:
- Heng's resnet34 classifier  (found here: https://www.kaggle.com/bibek777/heng-s-model-inference-kernel) ;sorry, didn't notice appropriate model in mlcomp kernel : /
- One more unet with resnext50_32x4d encoder (fold number 3)


What could work (it gave higher private score, but i didn't select that solutions):
- Efficientnet-b5/vgg16 encoders 
- More resnext50_32x4d encoders trained on different folds 


What didn't work:
- Additional augmentations for classification (vertical and horizontal flip) and segmentation (vertical flip)
- Thresholds higher than 0.5


Here is my code:
https://www.kaggle.com/shiron8bit/silver-frankenstein

Kudos to @hengck23 and @lightforever!
