# 5th place solution, maskrcnn, pseudo label and cellpose

Competition: sartorius-cell-instance-segmentation
Rank: #5
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298081

Congrats to all the winners, and thanks to Sartorius team and Kaggle hosted this competition, it’s very challenging and interest.

**Methods**

My pipeline can be described in the following figure.
[pipeline.png]
Step 1: MaskRCNN models were trained on Detectron2 using LIVECell_dataset_2021 without shsy5y cells.
Step 2: Finetune the MaskRCNN models on competition train dataset and LIVECell shsy5y cells.
Step 3: Predict the train_semi_supervised dataset and used them as pseudo labels.
Step 4: Finetune the models using LIVECellshsy5y cells + competition trainset + train_semi_supervised(pseudo)
Step 5: Predict the competition trainset and LIVECellshsy5y cells using the above models.
Step 6: Generate the flow-x, flow-y and semantic segmentation base on Step5 results using Cellpose.
Step 7: Train Cellpose model on LIVECell_dataset_2021 without shsy5y cells.
Step 8: Finetune Cellpose model using Step7 results as additional channels.
Step 9: Predict and post-process using Cellpose. The diameter was set to 19 and re-predict.

The MaskRCNN+Cellpose architecture is in the following figure.
[cellpose-arch.png]
**Results**
[results.png]
**Things tried but not worked**
1.Deep Watershed and it’s variants.
2.OmniPose 

**Things want to try if have more time**
1.Cellpose as additional heads of MaskRCNN.
2.Detection models using all kinds of datasets+UNET Singel Cell segmentation.


**Happy New Year!**
