# 16-th Solution: Mask-RCNN on Steroids

Competition: imaterialist-fashion-2019-FGVC6
Rank: #16
Source: https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/95249#latest-550784

*(Also known as Alex doesn't know how to code)*

After seeing first place's solution, I was shocked at how close I was to reaching the gold area... This competition has taught me a lot on hyper-parameter fine-tuning and architecture selection for instance segmentation, and the greatest (and most painful) of them all is the threshold for non-maximum-suppression- and I fell for it. This is my first serious attempt at aiming for grandmaster status, and though the outcome is not so satisfactory, for the most part I enjoyed the competition.

For my solution, it is based on mmdetection, an awesome codebase for detection related tasks. 

### I started with Hybrid Task Cascade with ResNeXt-101-64x4d-FPN backbone, with the following modifications:

1. DCN-v2 for part of the backbone

2. Removing mask fusion branch (since it takes too much memory and semantic information in this task is tricky to define)

3. Multi-scale training with flipping and resizing range (512,512) -&gt; (1333,1333)

4. Guided-anchoring RPN

5. Increasing initial mask branch output from 14 pixels to 28 pixels

### Then I modified the training progress a bit.

1. Default training schedule, 1 image per GPU for 4/8 GPUs

2. Changing mask loss from BCE to symmetric lovasz:

`def symmetric_lovasz(outputs, targets):
return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2`

3. Changing smooth L1 loss to GHM-R loss

4. Changing sigmoid-based losses to Focal Loss and GHM-C loss

### Then, for inference:

1. Used soft-nms ( 
#but I didn't fine-tune the threshold :(


2. Multi-scale testing in mmdetection wasn't implemented, so I wrote one myself that merge predictions in multiple scales in proposals, bboxes and masks. The following scales are used (with flipping): [(1333,1333), (1333,800), (800,1333), (900,600)]

3. Deleting masks with low confidence &amp; low pixels (with fine-tuned threshold)

4. Heuristic that make sure that the same pixel are not assigned to two instances w. the same ClassId

5. (I couldn't submit for the final minutes so I didn't have a chance to test this out) Binary classifier that predict if an object is fine-grained or not. Then, in final submission, if an instance is determined to be fine-grained, it is excluded. The rationale is as below:

  a) An included object that is fine-grained but the mask generated is not: 1 FP + 1 FN
  b) Simply not predicting that object: 1 FN

And for this network, it's a se-resnext50 based image classifier with 4-channel input (image + binary instance mask) and one-hot category fusion in the fc-layer. Honestly I could be doing a DAE before the classifier (from dirty testing mask to imagined clean train mask) but I ran out of GPUs and time.

### What didn't work:

1. Un-freezing bn / gn / synchronized bn

2. Parallelized testing

3. After-nms ensemble (like the one used in Top3)

4. Attribute classification (my local F1 was &lt; 0.1)

### Machines

I used a variety of rented cloud machines, from 1 * 2080 Ti, 1 * P6000 to 8 * 1080 Ti to 4 * P100. I've basically spent all of the prizes I earned in the Whales competition on this one, plus a few hundred bucks :(

For your reference:

1. For 1-image per GPU, you can fit an (1333,1333) image into an RTX 2080 Ti w.o. the enlargement of mask branch or GA-RPN

2. But if you want to add GA-RPN, only GTX 1080 Ti would suffice (the pitfall: **1080 Ti has a little bit more memory than it's RTX counterpart**)

3. My full setting fits barely in an P100.

### Final words

I have to say goodbye to kaggle for the following months due to internships and school-related stuffs, so it's a little discouraging to know that I missed the last chance this year to getting a solo gold by such a small margin. But I guess the journey and the things I've learnt is more important than the recognition itself. So I will come back with full strength after a small break. Also congratulations to the winners, you guys did a fantastic job!
