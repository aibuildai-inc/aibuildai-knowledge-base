# 15th place solution with Deep Supervision

Competition: human-protein-atlas-image-classification
Rank: #15
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77322

Here is a writeup on my team's approach to this challenge

Data: Initially we split the Kaggle dataset into 6 folds. After the external data was discovered, we incorporated it directly into the train part in every fold (so we had roughly 100k train images and 5k valid. images). We trained and predicted using resized 512x512 RGB/ RGBY images.

Models:  SE_ResNext50/101 and Inceptionv3/v4
We realized that multi-scale predictions are really crucial since the trace of a protein may appear in a very small region of the image. We solved this problem by applying the deep supervision trick. The chosen networks consist of multiple blocks of gradually decreasing resolutions; on top of these blocks, a small auxiliary classification network (Global Pooling + BN + Dense) was built. The final loss is an equal sum of these auxiliary losses. At inference time, we averaged all the auxiliary blocks' predictions and the final Dense layer predictions. For Inception networks, we picked 6 mixed blocks.  For ResNext networks, block 3, 4, 5 and a feature pyramid network outputs were used.

Loss: Pure binary cross entropy, no upsampling/downsampling rare classes. Focal loss didn't work well for us.

Threshold: 0.5 for several popular classes, rest 0.2

Final submission is a weighted average of 4 above models. Stacking gave a strong 5% boost on  local F1 score yet sucked both in public and private LB. My teammate also did extra postprocessing on submission file but we didn't go with it at the end since it gave slightly lower public LB score. Sadly, this post-processed submission would have given us another gold medal XD

Finally, huge congrats to all the winning teams and gold medalists XD
