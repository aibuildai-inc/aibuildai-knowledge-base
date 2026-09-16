# 9th Place Solution

Competition: blood-vessel-segmentation
Rank: #9
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475080

Thank you for host, I learned a lot from this competiton!
also special thanks to @hengck23 . I refer @hengck23 discussion topics many times.

# TL;DR
- MaxVit tiny
- xy, xz, yz inference
- heavy regularization
- probability threshold

# Solution
## Model
- MaxVit tiny

```
class SenUNetStem(nn.Module):
    def __init__(self, encoder_name="resnest26d",output_stride=32,
                 encoder_depth=5 , in_chans=1,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None, classes=1, activation=None):
        super(SenUNetStem, self).__init__()
        kwargs = dict(
            in_chans=in_chans,
            features_only=True,
            # output_stride=output_stride,
            pretrained=True,
            out_indices=tuple(range(encoder_depth)),
        )
        self.conv_stem = Conv2dReLU(in_chans, 16, 3, use_layernorm=False, padding=1)
        self.encoder = timm.create_model(encoder_name, **kwargs)
        self._out_channels = [
            32,
        ] + self.encoder.feature_info.channels()

        self.decoder = UnetDecoder(
            encoder_channels=self._out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1] + 16,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.n_time = n_time
        self.pickup_index = pickup_index

    def forward(self, x):
        B, C, H, W = x.shape
        h = (H//32)*32
        w = (W//32)*32
        x = x[:,:,:h,:w]
        stem = self.conv_stem(x)
        features = self.encoder(x)        
        features = [
            stem,
        ] + features

        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        masks = F.pad(masks,[0,W-w,0,H-h,0,0,0,0], mode='constant', value=0)
        
        return masks[:,0]

SenUNetStem(
            encoder_name="maxvit_tiny_tf_512.in1k",
            classes=1,
            activation=None,
        )
```

## Dataset
- kidney1 and 3 dataset.

## Training Tricks
I focus on regularization trick.
because this competiton have unstable cv, public is not related.
Moreover host shared public/Private LB information images, I hink it indicate unstable.

- EMA
- 50epochs
- AdamW(Weight Decay 1e-2)
- CutMix(until 25ep)
- MixUp(until 25ep)
- DiceLoss(smooth_factor=0.1)
- Heavy Augmentation 
```
    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightness(limit=0.1, p=0.7),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(blur_limit=3),
                ], p=0.4),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=1., p=1.0)
        ],p=0.2),
        A.ShiftScaleRotate(p=0.7, scale_limit=0.5, shift_limit=0.2, rotate_limit=30),
        A.CoarseDropout(max_holes=1, max_height=0.25, max_width=0.25),
        ToTensorV2(transpose_mask=True)
    ])
```
- Crop(512)

## Inference
Inference is xy, xz, yz axis, and crop 512, stride 256.

## Post-Process
Probability threshold(=sigmoid output). I didn't use percentile which method used many public notebook and past segmentation competiton(e.g. Volcano).
Because I checked percentile threshold in local cv, it's not stable. I didn't use it.

## Not worked
- Bigger models(maxvit base, small)
- large size inference(1024), 512 is enough for this competiton.
- Rotate90
- Pretrained Other volumes(kidney_2/kidney_1_volumes)

https://www.kaggle.com/code/tereka/simpleunet-xy-xz-yz-v2-nbp-b749ff/notebook
