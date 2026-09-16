# 11th place solution

Competition: sartorius-cell-instance-segmentation
Rank: #11
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298038

First of all congratz to the winners and thanks to Sartorius and Kaggle for hosting the competition !

Although this 11th place is a great finish, we're a bit disappointed since we spent a month at the top and the final weeks in the top 5. We knew we were not gonna win but this shake was quite a surprise for us, and we still haven't really understood why it happened. 

Our models were just better on Public than Private, there might be a small domain shift (more corts ? more small cells ? more large cells ? we will never now...) that emphasized a weakness of our pipeline.

Anyways, here are few interesting points of the solution that I found were worth mentioning, the full code is on GitHub : https://github.com/TheoViel/kaggle_sartorius 
Also, our inference code is here : https://www.kaggle.com/theoviel/sartorius-inference-final

### Validation Scheme

Two weeks before the end of the competition, we switched to a validation set-up we thought would be reliable : splitting on plate & well as done in the Livecell paper. 



This might be one of the reason we shake down ? Our best private LB (0.349, public 0.339) was our best CV before switching to the above scheme. Still, our final score of 0.348 is our best CV with this setup (public 0.342)

### Models

We only used machines with single RTX 2080 Ti so we had to be ingenious to be able to train on high resolution images and detect small cells. We used mask-rcnn based models and relied on mmdet but only for model definition and augmentations, the rest of the pipeline is hand-crafted, which made it more convenient for experimenting.




##### Main points
- Remove the stride of the first layer of the encoder to "increase the resolution" of the models without doing any resizing !
- Random Crops of size 256x256 for training
- Pretrain on Livecell
- 4000 iterations of finetuning on the training data
- Backbones : resnet50, resnext101_32x4, resnext101_64x4, efficientnet_b4/b5/b6
- Models : MaskRCNN, Cascade, HTC

##### Ensembling

We average predictions of different models & different flips at three stages, the stages are the boxes with thicker borders above.
- Proposals : For a given feature map output by the FPN, each of its pixel is assigned a score and a coordinates prediction by the convolutions. This is what we average. 
- Boxes : We re-use the ensembled proposal and perform averaging of the class predictions and coordinates for each proposal. We used 4 flip TTAs.
- Masks : starting with the ensembled boxes, we average the masks - before the upsampling back to the original image size.

This scheme doesn't really use NMS for ensembling which can be tricky to use. Hence we stacked a bunch of models. We used 6 models per cell type.

##### Post processing

- NMS on boxes using high thresholds, then NMS on masks using low thresholds
- Corrupt back the astro masks as we trained on clean ones (+0.002 LB)
- Small masks removal

We did a lot of hyper-parameters tweaking on CV : NMS thresholds, RPN and bbox_head params, confidence thresholds, minimum cell sizes, mask thresholds.

### Few more words

- Pseudo Labelling didn't really work for us, we used them in the finetuning phase with the original training data and progressively decayed their proportion.
- The RoiAlign layer from mmdet has implementation issues. Masks resulting from TTA appear shifted which hurt performances, especially when trying to use vertical flips. We had to shift the boxes by 0.5 to counter this. 

 I will probably add more stuff later, and fix the typos and all. Feel free to ask any questions  !
*Thanks for reading !*
