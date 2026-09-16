# 6th Place Solution - Transformer Tweaking Madness

Competition: asl-signs
Rank: #6
Source: https://www.kaggle.com/c/asl-signs/discussion/406537

Thanks to both, the organizers of this competition who offered a fun yet challenging problem as well as all of the other competitors - well done to everyone who worked hard for small incremental increases. 

Although I am the one posting the topic, this is the result of a great team effort, so big shoutout to @christofhenkel.

# Brief Summary

Our solution is a 2 model ensemble of a MLP-encoder-frame-transformer model. We pushed our transformer models close to the limit and implemented a lot of tricks to climb up to 6th place. 
I have 1403 hours of experiment monitoring time in April (that’s 48h per day :)).

**Update :** Code is available here : https://github.com/TheoViel/kaggle_islr

# Detailed Summary

## Preprocessing & Model

<a href="https://ibb.co/g3np4jQ">[ISLR-drawio-1]</a>

### Preprocessing
- Remove frames without fingers
- Stride the sequence (use 1 every n frames) such that the sequence size is `<= max_len`. We used `max_len=25` and `80` in the final ensemble
- Normalization is done for the whole sequence to have 0 mean and 1 std. We do an extra centering before the specific MLP. Nan values are set to 0

### Embedding
- 2x 1D convolutions (`k=5`) to smooth the positions
- Embed the landmark id and type (e.g. lips, right hand, ...), `embed_dim=dense_dim=16`

### Feature extractor
- One MLP combining all the features, and 4 for specific landmark types (2x hands, face, lips)
- Max aggregation for hands is to take into account that signers use one hand
- `dim=192`, `dropout=0.25`

### Transformer
- Deberta was better than Bert, but we had to rewrite the attention layer for it to be efficient
- To reduce the number of parameters, we use a smaller first transformer, and modify the output layer to upscale/downscale the features. This was key to enable blending 2 models
- `d_in=512`, `δ=256` for `max_len=25`, `δ=64` for `max_len=80`, `num_heads=16`, `dropout=0.05` for the first layer, `0.1` for the other 2
- Unfortunately, we did not have any luck with using a pre-trained version of Deberta, for example by importing some of the pretrained weights

## Training strategy
<a href="https://imgbb.com/">[ISLR-Train-drawio-1]</a>

### Augmentations:
- Horizontal Flip (`p=0.5`)
- Rotate around `(0, 0, 0)` by an angle between -60 and 60°  (`p=0.5`)
- Resizing by a factor in `(0.7, 1.3)` (`p=0.5`). We also allow for distortion (`p=0.5`)
- Crop 20% of the start or end (`p=0.5`)
- Interpolate to fill missing values (`p=0.5`)
- Manifold Mixup [[1]](https://arxiv.org/abs/1806.05236) (scheduled, `p=0.5 * epoch / (0.9 * n_epochs)`) : random apply mixup to the features before one of the transformer layer
- Only during the first half of the training, since it improved convergence
  - Fill the value of the missing hand with those of the existing one (`p=0.25`)
  - Face CutMix : replace the face landmarks with those of another signer doing the same sign (`p=0.25`)

### Training
- 100 epochs, `lr=3e-4`, 25% warmup, linear schedule
- Cross entropy with smoothing (`eps=0.3`)
- `weight_decay=0.4`, `batch_size=32` (x8 GPUs)
- OUSM [[2]](https://arxiv.org/pdf/1901.07759.pdf), i.e. exclude the top k (`k=3`) samples with the highest loss from the computation
- Mean teacher [[3]](https://arxiv.org/abs/1703.01780) & Knowledge Distillation (see image above). We train 3 models at the same time, and use the distilled one for inference
- Model soup [[4]](https://arxiv.org/abs/2203.05482) of the last 10 epochs checkpoints

## Experiments 

<a href="https://ibb.co/MV343Xx">[cvlb]</a>

###  Validation strategy
We use a Stratified 4-fold, grouped by patient. We had a great CV - LB correlation (see figure), and our best model achieved CV 0.749 - Public 0.795 - Private 0.877. We submitted single models trained on the full dataset, and 2 models for ensembles.

###  What worked & reported improvements
Starting from early April :
- Baseline : MLP + Bert : LB 0.72
- Deberta instead of Bert : CV +0.015  **-> LB 0.73**
- Improve Feature extractor : CV +0.01  **-> LB 0.74**
- Flip augmentation : CV +0.015  **-> LB 0.75**
- Increase model size : CV +0.006 
- Two stage training + interp aug : CV + 0.002 ?
- Increase model size  : CV +0.005  **-> LB 0.76**
- Deberta Rework : CV +0.005   **-> LB 0.77**
- OUSM : CV +0.003
- Mean Teacher : CV +0.005
- Ensemble two distilled models : +0.005 **-> LB 0.78**
- Conv layers : CV +0.001
- Mish [[5]](https://arxiv.org/abs/1908.08681) activation instead of ReLU/LeakyReLU : CV +0.001
- Warmup 0.1 -> 0.25 : CV +0.001
- Model soup : CV +0.001
- [Nobuco](https://github.com/AlexanderLutsenko/nobuco) instead of onnx : 45ms/it -> 38ms/it, this enabled to improve the ensemble 
- Weight decay 0 -> 0.4 : CV +0.004
- Truncation aug : CV +0.001
- Manifold mixup : CV +0.002
- Different input, model and teacher size for ensemble : CV +0.005 **-> LB 0.79**
- Centering before the MLP layers : CV +0.004  **-> LB 0.795  (final)**

###  What did not work (for us)
- Unfortunately we could not make CNNs work
- GCNs, or other architectures that are “sota” for ASL
- Heng’s architecture, although others had great results with it
- Getting a successful 3-model sub
- Relabel the data to reduce label noise
- Other augmentations such as noise, dropout, drop frame, shift
- Dropout scheduling, custom lr per layer
- Pretraining part of the model
- Handcrafted graph features (adjacency matrices, edge features)
- Adding more landmarks (eyes, eyebrows)


*Thanks for reading !*
