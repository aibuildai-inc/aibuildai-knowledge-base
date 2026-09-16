# 8th place solution ... close but no cigar

Competition: asl-signs
Rank: #8
Source: https://www.kaggle.com/c/asl-signs/discussion/406411

Here is a quick overview of the 8th place solution. 

3 transformers models, 2 layers each (384 hidden, 512 hidden ffn), with an ffn encoder (512->384), trained from scratch. LR 8e-4 with cosine schedule trained for ~300 epochs, dropout 0.1, batch size 1024, label smoothing 0.1. Using hands, lips and pose (above waist only). On one transformer all pose and a subset of lips were used for diversity.  

**Augmentations**
- most important was sequence cutout. On each sample, and each body part (left hand, right hand, lips, pose) with a 0.4 proba convert to nan 5 random slices of 0.15 x SequenceLength. It was hard to overfit with this in.
- mirror left
- random rotate. 

**Preprocessing**
- Linear interpolation of longer sequences to max length of 96.  
- Normalise each body part, using min max - I found this better than mean/std. In one model I used mean/std for diversity. 
- Create time shift delta features on a subset of points, using time shifts of `[1, 2, 3, 4, 6, 8, 12, 16]`. It was important to forward fill NANs as opposed to 0-fill. 
- Angle features between xyz position, and xy positions, on a few different joints - corner of mouth, hand, arms. 
- Some point to point distances, interestingly this did not help much. 

**Tflite**
Training was all in pytorch. Converting to tflite, the feature preprocessing was rewritten in tensorflow, and the base models were converted via onnx to tf (this turned out to be a mistake from looking at #2 solution, I should have rewritten the transformer encoder). It was a great opportunity to learn tensorflow. I like it. 
A funny thing... doing the following on the pytorch model `model = model.half().float()` before converting to onnx, gave a good speed up in the final tflite inference. I tried quantizing the pytorch model and it did not further increase the speed. 

In the last week I found in preprocessing, normalising points across the whole sequence channel wise gave a good lift. Particularly what I tried to do was for each body part get the channel wise range (max-min), then get the average range over all frames, and use this to normalise the time deltas and the raw coordinates - as opposed to normalising each frame independently. Every time I submitted this it gave a very low score after ~25 mins. I rewrote it in different ways a number of times and it worked with tfilte in kaggle kernels but not when submitted. It was very frustrating at the end. However, I do not think the boost would have brought me into the prize range which was the goal 🤑

**Failed attempts**
- Mixup worked (~~as mentioned by #9 team~~ worked for #9 team), I tried it on the embeddings and it worked ok, ~~but I did not finetune the result on the data without mixup, which probably would have helped~~. See comments below on mixup.
- CNNs with mixup - tried efficientnet and edgenext. I tried on raw points and it did not work, looked like I had an incorrect normalisation from #2 solution. It worked well with an FFN encoder, but was too slow in inference. Looks like I missed to rewrite it into tf, or train in tf.
