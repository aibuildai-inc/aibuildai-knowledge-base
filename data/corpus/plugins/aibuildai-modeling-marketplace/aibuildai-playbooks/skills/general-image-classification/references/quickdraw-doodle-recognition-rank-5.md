# 5th place solution

Competition: quickdraw-doodle-recognition
Rank: #5
Source: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73708

First of all, congrats everyone with the end of the competition. It was an exciting experience, and we want to share our approach.

**Handling the data**

Both simplified and raw data were used. To be able to read any random image, all strokes from CSV files were separated to one image per binary file. It took about 400GB on SSD, but it allowed to start different experiments very fast.

**Our models**

In total we trained 3 main models:

 1. Se-Resnext50
 2. DPN-92
 3. Se-Resnext101

All of them were pretrained on imagenet. We used different image sizes, 128 -&gt; 192 -&gt; 224 -&gt; 256. It was clear almost from the beginning - the bigger image size -  the bigger score in both local validation and public lb. It was hard to train 256px due to limited GPU resources. Using fit predict and 128px image it was pretty straightforward to get 0.944 public LB.

In the middle of the competition after merging with @firenero, we had ~ 0.948 public lb score. To move forward, it was essential to use time information which only exists in the full dataset. We encoded each stroke using 3 channels:

 - Delay value scaled to 0-255.
 - Draw time per stroke scaled to 0-255
 - Number of strokes scaled to 0-255

It gave a significant boost in local validation and gave us ~0.951 public LB.
Another important thing is batch size; we tuned all our models with huge batch size increasing it with each snapshot up to 10K.

@firenero is going to tell more about the final phase and how we achieved 0.953.
