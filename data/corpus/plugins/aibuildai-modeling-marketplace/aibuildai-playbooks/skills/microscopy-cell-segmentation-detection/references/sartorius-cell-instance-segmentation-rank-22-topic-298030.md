# 22th place solution / for English edition and 中文版本

Competition: sartorius-cell-instance-segmentation
Rank: #22
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298030

Thanks to my teammates in this comp ! @wgz123 @deeeeeeeplearning @shajiayu They do a lot in  this comp!

#English edition
`SOTA`: Mask-RCNN
`backbone`: SwinTransformer
`anchor_size`: mutil anchor size for different sortation
`anchor_ratio`: different ratio for different sortation
`pre-trained model`: by using outer data provided by this comp
`data augmentations`: Resize(), Pad(), Grid(), RandomFlip()
`TTA`: Resize(), Pad(), RandomFlip()
`Lr-plan`: CosineAnnealing
`Four models ensemble`: One model for classifier and the rest for segmentation
`swa`: Stochastic Weights Averaging
`score_thr`: min score for each instance
`min_pixel`: min pixel for each instance
`fix overlap`: fix overlap by the index

#中文版本
`SOTA`：Mask-RCNN
`骨干网`：SwinTransformer
`框大小策略`：不同类别采用不同的框大小
`宽高比策略`：统计标注种类宽高比分布
`预训练`：官方提供的外部数据进行预训练
`数据增强`：Resize(), Pad(), Grid(), RandomFlip()
`TTA`：Resize(), Pad(), RandomFlip()
`学习率策略`：余弦学习率 
`四模型集成`：一模型分类三模型分割
`随机加权平均`：swa
`卡置信度阈值`：min_score
`卡最小像素阈值`：min_pixel
`去除重叠`：按照索引顺序去除


Second comp in kaggle! keeping fighting!
