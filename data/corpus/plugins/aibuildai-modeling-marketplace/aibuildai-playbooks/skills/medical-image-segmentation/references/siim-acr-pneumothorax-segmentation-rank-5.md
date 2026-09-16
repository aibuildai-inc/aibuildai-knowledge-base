# 5th place solution && code updated

Competition: siim-acr-pneumothorax-segmentation
Rank: #5
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107603

Congratulations to all !
My solution is based on semi-supervision, and I added a 2-class classifier for pneumothorax in the network.

Network : Unet with Aspp
Backbone : se50 &amp; se101
Image size: (1024, 1024)
Optimizer : Adam
Loss : 1024 * BCE(results, masks) + BCE(cls, cls_target)
Semi-supervision:  mean-teacher[1-2]  with NIH Dataset （0.874 ---&gt; 0.880）

Scores: 
	stage1  0.8821
	stage2  0.8643

[1] https://github.com/CuriousAI/mean-teacher
[2] https://arxiv.org/pdf/1703.01780.pdf

code : https://github.com/earhian/SIIM-ACR-Pneumothorax-Segmentation-5th
