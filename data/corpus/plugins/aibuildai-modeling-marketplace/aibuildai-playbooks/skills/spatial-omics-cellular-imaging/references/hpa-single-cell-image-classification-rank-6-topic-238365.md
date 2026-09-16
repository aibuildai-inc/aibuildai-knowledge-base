# What a ride! Parts of 6th place solution

Competition: hpa-single-cell-image-classification
Rank: #6
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238365

I put all of my heart into this competition, and I need to start by thanking my wife for putting up with me in the last couple of months - buying a DL rig, waking up or going to sleep at crazy hours, putting my rig on car front seat on a weekend trip, and many more… all while going through Covid (fortunately mildly) and trying to entertain our Kaggle-famous 4yo future scientist while quarantined at home :) 

Thanks to the team members - @dschettler8845 and @felipebihaiek for partnering early on, experimenting together and discussing hundreds of ideas, @scusywxy and @zehuigong for the partnership and your lead to figure out the engineering details so that our combined solution finally worked. We had diverse approaches which gave us a boost (2 minutes late for the money zone!). I will only share my contributions here, a more comprehensive summary of our total approach will follow. 

Thanks to Kaggle and the hosts for this competition!

# Model or Data-Centric Approach?

We started the competition with single-cell classification combined with HPA CellSegmentator, and I believed initially this will be the key to the competition. I planned a data centric approach, including pseudo labelling (it gave me a small boost early on) and iterative fixing of most-confused labels. I wanted to use a fastai widget for that, but afaik it doesn’t support multilabel classification unfortunately, so experimented with some other tools, but finally gave up when we realized labelling the cells is not very straightforward. 

We had accurate data to train image level models, so the question was how we can use those models to predict cells? After many experiments, we figured out two approaches that led us first to .549 public lb score, and gave us a 0.02 boost after blending with @scusywxy and @zehuigong models. 

# GAPMASK Inference

We train a regular image-level model. At inference, we modify the model architecture so that it takes two inputs - image and cell-mask. The image goes through the network alone until the GAP layer. At that point, we do element wise multiplication of the image activations and cell mask. From that point, an image gets expanded into a batch of single-cell images, and we apply the model head to that entire batch. 

[Gap-Mask-1]

We had two variations of this architecture - one for global average and max concat pooling, one for attention pooling. 

This is the code for our densenet model: 

```
    def forward(self, images, masks):
        …
        e5 = F.relu(e5,inplace=True)
        # GAP MASK STARTS HERE:
        e5 = F.interpolate(e5, scale_factor=2, mode='bilinear') # increase grid size to 48x48
        x = e5.permute(0,2,3,1) # [1, 48, 48, 1024]
        x = x.squeeze() # [48, 48, 1024]
        msk = masks.squeeze() # [15, 48, 48]
        res = msk[...,None] * x[None,...] # [15, 48, 48, 1024]
        res = res.permute(0,3,1,2)
        # NOW APPLY THE MODEL HEAD WITH THE MASK DIMENSION AS THE BATCH DIMENSION
        x = torch.cat((nn.AdaptiveAvgPool2d(1)(res), nn.AdaptiveMaxPool2d(1)(res)), dim=1)
        x = x.view(x.size(0), -1)
	…
        x = self.logit(x)
        return x

```
And this is the implementation for inception with attention: 
[Gap-Mask-2]

```
    def forward(self, x, masks):
        …
	# GAP MASK STARTS HERE (dimensions reflect an image with 15 cells):
        logits = self.last_linear(features_b)
        logits = F.interpolate(logits, scale_factor=3, mode='nearest')
        logits_attention = self.attention(features_b)
        logits_attention = F.interpolate(logits_attention, scale_factor=3, mode='nearest')
        logits_attention = logits_attention.view(-1, self.num_classes, self.attention_size * 3 * self.attention_size * 3)
        attention = F.softmax(logits_attention, dim=2)
        attention = attention.view(-1, self.num_classes, self.attention_size * 3, self.attention_size * 3)
        attention = attention.permute(0,2,3,1) # [1,66,66,19])
        msk = masks.squeeze() # [15,66,66]       
        attention = msk[...,None] * attention # [15,66,66,19]
        attention = attention.permute(0,3,1,2) # [15,19,66,66]
        attention = attention.view(-1, self.num_classes, self.attention_size * 3 * self.attention_size * 3) #[15, 19, 66*66]
        attention = attention / (attention.sum(2).unsqueeze(-1) + 1e-7)
        attention = attention.view(-1, self.num_classes, self.attention_size * 3, self.attention_size * 3) #[15,19,66,66]
        logits = logits * attention
        return logits.view(-1, self.num_classes, self.attention_size * 3 * self.attention_size * 3).sum(2).view(-1, self.num_classes) # torch.Size([1, 19])
```

# GRIDIFY Inference

When we started with single cell tiles, we resized each cell to the same size, e.g. 128x128. The problem is that this changed cell resolutions, some
got shrunk and some expanded. Inspired by [this paper](https://arxiv.org/pdf/1906.06423.pdf), I looked for an approach that would line up the features model sees while training on full images, with features seen by the model when running inference on single cells. 

The initial gridify approach was to take a single cell crop, copy-paste that crop with multiple augmentations into a 2048x2048 template, and then resize in the same way as during training. Finally, we converged on the following approach: 
- Take 4x 512x512 crop from full image (independent of size) from each corner of the single cell
- Put those 4 crops into a single 1024x1024 image
- Resize to maintain same resolution as training. Eg. for the model trained on 768 size (from 2048), we resize into 384x384 cell tile

[gridify]

# Models

We used three models from HPA 2018 winning solutions: 
- Densenet trained on size 768 from bestfitting
- Densenet trained on size 1536 from bestfitting 
- Inceptionv3 trained on size 1024 from pudae

All models were fine-tuned with original loss functions (focal-lovasz-logloss for densenet, focal for inception), starting with the 2018 weights. We froze the model head for 1 epoch, then trained for 3-6 epochs on train + public data (excl. classes 0 and 16) with cosine annealing. 

# Validation

Due to potential leakage from similar images (same plate) across folds, we run all images through metric learning model (from bestfitting’s 2018 solution) and cluster similar images together with UMAP/DBSCAN. We treat the clusters as groups and divide data in folds based on group stratified multilabel approach. 

# Postprocessing

We reduce probabilities for cells based on negative signal, this gave us a small boost (~0.003): preds[:,:18] = preds[:,:18] * (1 - preds[:,18]).unsqueeze(-1)

# Tried but didn't work

- Pseudolabels
- Multi-instance learning (treat each image as a bag of cells, train with bag of cells, predict on single cells). This was the same architecture as shared by Guanshuo Xu in his #8 solution. After analyzing what I did wrong, my conclusion is that I used too many cells per image. After reducing number of cells from 48 to 8, the results for my previously failed experiments improved significantly.
- Using rectangle-shape images during inference
- BagNet [paper](https://arxiv.org/pdf/1904.00760.pdf)
- ...
