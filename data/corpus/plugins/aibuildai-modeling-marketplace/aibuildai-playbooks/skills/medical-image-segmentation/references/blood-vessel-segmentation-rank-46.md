# Rank 46 Solution - keep the CV discipline

Competition: blood-vessel-segmentation
Rank: #46
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475050

## Our solution

Thanks to the organisers for the learning opportunities and thanks @shashwatraman for helping me in this comp.



**Submission Picking rule:**
We played against the shakeup by keeping the CV simple and the one that made the most sense for us: validation on kidney_3 dense. We didn't trust the LB at all in  this competition and decided to pick the submission that followed the following criteria:
- Tiling Model - avoid weird shapes of the private LB
- Used interpolation augs (resizing and random resize crop) - resolution shift
- Last epoch weights (no checkpointing) - avoid overconfident CV


**After that, the training protocol is very simple and there is no post/preprocessing done.**
- 2D unet
- Tiling as mentionned before (384x384)
- maxvit_small_384 single model (no ensembling) (we didn't have time xd)
- 15 epochs
- volume norm
- light augs 
- Low threshold: 0.1


**Takeaways:**
Most of the things we tried in the comp didn't work, the things that worked best on CV was the 2.5d and 3d approaches, but I didn't want to trust that the z axis resolution was going to allow such models.
I thought about distilling those preds to a 2D model but i didn't have the time for it.
I would love to participate in another 3D comp where everything is more stable and where I can use a bit more imagination than this one, where most of the score came from having the discipline of not looking at the LB. 
Even if I'm proud of our CV, our best submission scored 0.615, and if I really stuck to my rules, I would have picked it, but the LB for it was so low I did coward away from taking it, lesson learned.
