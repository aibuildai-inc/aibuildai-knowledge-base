# Second Place Solution

Competition: recursion-cellular-image-classification
Rank: #2
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110457

Thanks to Recursion and Kaggle for sponsoring this very interesting competition and congratulations to all that went through the journey.  We learned a lot and enjoyed the entire process.  Looking forward to learning from other teams’ solutions.

### Input and preprocessing:
- 6 channel input, 
- per image standardization (minus the mean and divide by the standard deviation), 
- random crop 384x384,
- random flip, 
- random rotation multiple of 90 degrees.  

### Modeling:
We modified ResNet to limit the receptive field size of the output, as we suspect it is the individual cells and their immediate neighbors that contain the most discriminating information.  Here are the list of things we modified from the vanilla ResNet:
- Fewer blocks. ResNet typically has 4 chunks of blocks, some of our models only has 2 chunks.
- More 1x1 conv blocks 
- Average pooling from lower blocks, concatenated with average pooling from higher blocks
- Remove the immediate max-pooling after the first convolution
- Replace the 1x1 convolution with 3x3 convolution in the shortcut layer. This increased the smoothness of the test accuracy during training, but only increased the final testing accuracy slightly.  

We used all the negative and positive controls (including those in the test plates) as part of the training set.  The output is 1139x4 logits.

### Loss:
We used the [ArcFace loss posted by bestfitting] (https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109).  We made the `gamma` parameter adjustable.  We were struggling to get ArcFace to converge initially until we tuned the gamma parameter.  Our settings are `gamma = 0.2, m = 0.4, s = 32`

### Optimizer:
Adam optimizer with the following schedules:
``` python
0-20: linear warm up to 3e-3
20-60: 3e-3
60-80: 9e-4
80-90: 3e-4
90-100: 3e-5
100-120: 3e-6
```

### Post-processing:
- Center the embeddings by plate.
- Average the embeddings from both sites to obtain per-well embedding.
- For each cell line, obtain train center embeddings by averaging together the siRNA embeddings.
- Compute cosine-similarity of each well’s embedding to train center embeddings. 
- Use LSA to compute label assignment based on the 277 leak.

### Pseudo-labeling:
An ensemble of 5 models achieved a public LB score of 0.993 and a private LB score of 0.9957 without pseudo-labeling (single model 0.990 and 0.9947).  We then collected all our public LB 0.990+ predictions and identified 327 examples that were not consistent.  All the test predictions not in this set of 327 were then used as pseudo-labels.  An ensemble of models trained on this pseudo-label set achieved a public LB of 0.997, and a private LB of 0.9967.  We also tried iteratively adding more pseudo-labels to the training set (500, 700, 900 per experiment), but it did not improve our public LB score.
