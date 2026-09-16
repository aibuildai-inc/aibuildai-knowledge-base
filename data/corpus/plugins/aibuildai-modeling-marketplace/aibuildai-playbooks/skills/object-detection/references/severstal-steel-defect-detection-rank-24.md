# 24th place training framework using pytorch

Competition: severstal-steel-defect-detection
Rank: #24
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/116575

We got 24th place in Steel Defect Detection. When we played in this competetion, we found that there is few good semantic segmentation training framework in pytorch which made it a hard time to do experiments, so we developed a library which mimiced [mmdetection](https://github.com/open-mmlab/mmdetection).

Our implementation is named [vedaseg](https://github.com/Media-Smart/vedaseg), it contains fpn, unet, deeplabv3plus, deeplabv3, pspnet, etc. This implementation is a modular semantic segmentation library which is flexible and extensible. We can implement a new semantic segmentation model like building a lego toy.
