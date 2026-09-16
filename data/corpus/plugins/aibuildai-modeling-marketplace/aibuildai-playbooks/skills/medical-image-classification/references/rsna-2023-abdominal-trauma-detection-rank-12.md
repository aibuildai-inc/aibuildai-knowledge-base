# 12th place solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #12
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447539

Thanks kaggle and the hosts for this interesting competition, 
Big thanks to kagglers out there for their great ideas and engaging descussions.
Thanks a lot as well to my great teammate @siwooyong 

# Summary

Our solution is an ensemble of one stage approach without segmentation and two stage approach with segmentation.

## One stage approach (public LB = 0.4Thanks kaggle and the hosts for this interesting competition, 
Big thanks to kagglers out there for their great ideas and engaging descussions.
Thanks a lot as well to my great teammate @siwooyong 

# Summary

Our solution is an ensemble of one stage approach without segmentation and two stage approach with segmentation.

## One stage approach (public LB = 0.5, private LB = 0.45)

### Data Pre-Processing
If the dimension for the image is greater than (512, 512), we cropped the area with a higher density of pixels to get a (512, 512) image, then the input is resized to (96, 256, 256) for each serie following the same preprocessing steps that were used by hengck23 in his great [notebook](https://www.kaggle.com/code/hengck23/lb0-55-2-5d-3d-sample-model)

### Model : resnest50d + GRU Attention
We tried to predict each target independently from the others so we have 13 outputs

```python
class RSNAClassifier(nn.Module):
    def __init__(self, model_arch, hidden_dim=128, seq_len=3, pretrained=False):
        super().__init__()
        self.seq_len = seq_len
        self.model_arch = model_arch
        self.model = timm.create_model(model_arch, in_chans=3, pretrained=pretrained)


        cnn_feature = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)


        self.spatialdropout = SpatialDropout(CFG.dropout)
        self.gru = nn.GRU(cnn_feature, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.mlp_attention_layer = MLPAttentionNetwork(2 * hidden_dim)
        self.logits = nn.Sequential(
            nn.Linear(2 * hidden_dim, 13),
        )
    def forward(self, x):
        bs = x.size(0)
        x = x.reshape(bs*self.seq_len//3, 3, x.size(2), x.size(3))
        features = self.model(x)
        features = self.pooling(features).view(bs*self.seq_len//3, -1)
        features = self.spatialdropout(features) 
        # print(features.shape)
        features = features.reshape(bs, self.seq_len//3, -1) 
        features, _ = self.gru(features)            
        atten_out = self.mlp_attention_layer(features) 
        pred = self.logits(atten_out)
        pred = pred.view(bs, -1)
        return pred
```

### Augmentation
 * Mixup
 * Random crop + resize
 * Random shift, scale, rotate
 * shuffle randomly the indexes of the sequence, but respecting the same order and keeping the dependency between each three consecutive images:
```python
	inds = np.random.choice(np.arange(1, 96-1), 32, replace = False)
	inds.sort()
	inds = np.stack([inds-1, inds, inds+1]).T.flatten()
	image = image[inds]

```
Loss : BCEWithLogitsLoss
scheduler : CosineAnnealingLR
optimizer : AdamW
learning rate : 5e-5

### Postprocessing 
We simply multiplied the output by the weights of the competition metric :
```python
preds.loc[:, ['bowel_injury', 'kidney_low', 'liver_low', 'spleen_low']] *= 2
preds.loc[:, ['kidney_high', 'liver_high', 'spleen_high']] *= 4
preds.loc[:, ['extravasation_injury']] *= 6
```

## Two stage approach (public LB = 0.45, private LB = 0.43)
### stage1 : Segmentation

Model : regnety002 + unet

Even with only 160 of 200 data (1th fold) used as training data, the model has already shown good performance.

```python
class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
   
        self.n_classes = len(
            [
                'background',
                'liver',
                'spleen',
                'left kidney',
                'right kidney',
                'bowel'
            ])
        in_chans = 1
        self.encoder = timm.create_model(
            'regnety_002',
            pretrained=False,
            features_only=True,
            in_chans=in_chans,
        )
        encoder_channels = tuple(
            [in_chans]
            + [
                self.encoder.feature_info[i]["num_chs"]
                for i in range(len(self.encoder.feature_info))
            ]
        )
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=self.n_classes,
            activation=None,
            kernel_size=3,
        )

        self.bce_seg = nn.BCEWithLogitsLoss()

    def forward(self, x_in):
        enc_out = self.encoder(x_in)

        decoder_out = self.decoder(*[x_in] + enc_out)
        x_seg = self.segmentation_head(decoder_out)

        return nn.Sigmoid()(x_seg)
```

### stage2 : 2.5DCNN 

#### Data Pre-Processing:
We used the segmentation logits obtained from stage1 to crop livers, spleen, and kidney, and then resized each to (96, 224, 224). 
(We use 10-size padding when we crop the organs with segmentation logits)
In addition, full ct data not cropped is resized to (128, 224, 224) and a total of four inputs are put into the model (full_video, crop_liver, crop_spleen, crop_kidney)

#### Model : regnety002 + transformer

We initially used a custom any_injury_loss function, but found that it did not improve the performance.  For the model input channel, we experimented with different values, including 2, 3, 4, and 8. 
We found that a channel size of 2 performed the best, we also initially tried using a shared CNN and transformer model for all organs,  but found that separate CNN and transformer models for each organ performed better.  we also experimented with increasing the size of the CNN (using ConvNeXt and EfficientNet models), but this resulted in a decrease in performance.  Therefore, we used the RegNet002 model, which is a smaller CNN model.

```python
class FeatureExtractor(nn.Module):
    def __init__(self, hidden, num_channel):
        super(FeatureExtractor, self).__init__()

        self.hidden = hidden
        self.num_channel = num_channel

        self.cnn = timm.create_model(model_name = 'regnety_002',
                                     pretrained = True,
                                     num_classes = 0,
                                     in_chans = num_channel)

        self.fc = nn.Linear(hidden, hidden//2)

    def forward(self, x):
        batch_size, num_frame, h, w = x.shape
        x = x.reshape(batch_size, num_frame//self.num_channel, self.num_channel, h, w)
        x = x.reshape(-1, self.num_channel, h, w)
        x = self.cnn(x)
        x = x.reshape(batch_size, num_frame//self.num_channel, self.hidden)

        x = self.fc(x)
        return x

class ContextProcessor(nn.Module):
    def __init__(self, hidden):
        super(ContextProcessor, self).__init__()
        self.transformer = RobertaPreLayerNormModel(
            RobertaPreLayerNormConfig(
                hidden_size = hidden//2,
                num_hidden_layers = 1,
                num_attention_heads = 4,
                intermediate_size = hidden*2,
                hidden_act = 'gelu_new',
                )
            )

        del self.transformer.embeddings.word_embeddings

        self.dense = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.transformer(inputs_embeds = x).last_hidden_state

        apool = torch.mean(x, dim = 1)
        mpool, _ = torch.max(x, dim = 1)
        x = torch.cat([mpool, apool], dim = -1)

        x = self.dense(x)
        x = self.activation(x)
        return x

class Custom3DCNN(nn.Module):
    def __init__(self, hidden = 368, num_channel = 2):
        super(Custom3DCNN, self).__init__()

        self.full_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.kidney_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.liver_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.spleen_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)

        self.full_processor = ContextProcessor(hidden=hidden)
        self.kidney_processor = ContextProcessor(hidden=hidden)
        self.liver_processor = ContextProcessor(hidden=hidden)
        self.spleen_processor = ContextProcessor(hidden=hidden)

        self.bowel = nn.Linear(hidden, 2)
        self.extravasation = nn.Linear(hidden, 2)
        self.kidney = nn.Linear(hidden, 3)
        self.liver = nn.Linear(hidden, 3)
        self.spleen = nn.Linear(hidden, 3)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, full_input, crop_liver, crop_spleen, crop_kidney, mask, mode):
        full_output = self.full_extractor(full_input)
        kidney_output = self.kidney_extractor(crop_kidney)
        liver_output = self.liver_extractor(crop_liver)
        spleen_output = self.spleen_extractor(crop_spleen)

        full_output2 = self.full_processor(torch.cat([full_output, kidney_output, liver_output, spleen_output], dim = 1))
        kidney_output2 = self.kidney_processor(torch.cat([full_output, kidney_output], dim = 1))
        liver_output2 = self.liver_processor(torch.cat([full_output, liver_output], dim = 1))
        spleen_output2 = self.spleen_processor(torch.cat([full_output, spleen_output], dim = 1))

        bowel = self.bowel(full_output2)
        extravasation = self.extravasation(full_output2)
        kidney = self.kidney(kidney_output2)
        liver = self.liver(liver_output2)
        spleen = self.spleen(spleen_output2)


        any_injury = torch.stack([
            self.softmax(bowel)[:, 0],
            self.softmax(extravasation)[:, 0],
            self.softmax(kidney)[:, 0],
            self.softmax(liver)[:, 0],
            self.softmax(spleen)[:, 0]
        ], dim = -1)
        any_injury = 1 - any_injury
        any_injury, _ = any_injury.max(1)
        return bowel, extravasation, kidney, liver, spleen, any_injury

```

#### Augmentation

```python
class CustomAug(nn.Module):
    def __init__(self, prob = 0.5, s = 224):
        super(CustomAug, self).__init__()
        self.prob = prob

        self.do_random_rotate = v2.RandomRotation(
            degrees = (-45, 45),
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            expand = False,
            center = None,
            fill = 0
        )
        self.do_random_scale = v2.ScaleJitter(
            target_size = [s, s],
            scale_range = (0.8, 1.2),
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            antialias = True)

        self.do_random_crop = v2.RandomCrop(
            size = [s, s],
            #padding = None,
            pad_if_needed = True,
            fill = 0,
            padding_mode = 'constant'
        )

        self.do_horizontal_flip = v2.RandomHorizontalFlip(self.prob)
        self.do_vertical_flip = v2.RandomVerticalFlip(self.prob)
    def forward(self, x):
        if np.random.rand() < self.prob:
            x = self.do_random_rotate(x)

        if np.random.rand() < self.prob:
            x = self.do_random_scale(x)
            x = self.do_random_crop(x)

        x = self.do_horizontal_flip(x)
        x = self.do_vertical_flip(x)
        return x
```

Loss : nn.CrossEntropyLoss(no class weight)
scheduler : cosine_schedule_with_warmup
optimizer : AdamW
learning rate :2e-4

#### Postprocessing
We multiplied by the value that maximizes the validation score for each pred_df obtained for each fold.

```python
weights = [
    [0.9, 4, 2, 4, 2, 6, 6, 6],
    [0.9, 1, 4, 3, 2, 5, 5, 6],
    [0.2, 3, 2, 1, 2, 4, 2, 6],
    [0.5, 2, 2, 2, 2, 2, 6, 6],
    [1, 2, 3, 2, 6, 3, 6, 5]
]

y_pred = pred_df.copy().groupby('patient_id').mean().reset_index()

w1, w2, w3, w4, w5, w6, w7, w8 = weights[i]

y_pred['bowel_injury'] *= w1
y_pred['kidney_low'] *= w2
y_pred['liver_low'] *= w3
y_pred['spleen_low'] *= w4
y_pred['kidney_high'] *= w5
y_pred['liver_high'] *= w6
y_pred['spleen_high'] *= w7
y_pred['extravasation_injury'] *= w8

y_pred = y_pred ** 0.8

```

### Reference
[RSNA 2022 1st place solution](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607)
[RSNA 2022 2nd place solution](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/365115)
[RSNA 2022 5th place solution](https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place)
@hengck23's [descussion](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/435053)

## Code :
 https://github.com/siwooyong/RSNA-2023-Abdominal-Trauma-Detection
Inference notebook : https://www.kaggle.com/code/ahmedelfazouan/rsna-atd-channel2-512-inference-ensemble?scriptVersionId=146616538
