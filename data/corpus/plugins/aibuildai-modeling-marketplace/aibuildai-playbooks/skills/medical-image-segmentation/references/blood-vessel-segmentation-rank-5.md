# 5th place solution - 3D interpolation is all you need (updated with code)

Competition: blood-vessel-segmentation
Rank: #5
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475288

First of all, I want to express my deepest gratitude to the organizers of this competition. The reason I mostly compete in CV competitions is because I love it. Especially in medical competitions such as this one. So I'm really happy we managed to work with such a great technology (resolution is insane). 

### Validation 
Initially I thought it's gonna be a challenge validation-wise, since we simply don't have enough data for reliable validation. To make the validation without leakage and similar to test, I decided to train my pipelines in 2 ways: 
- Take kidney_1 as base for trainings, kidney_3 for validation
- Take kidney_3 as base for trainings, kidney_1 for validation

Since test is densely annotated, I wanted to compute metrics only on densely annotated kidneys, which eliminated kidney_2 from the discussion. 

### Data
I always believe that data is the key. So I tried my hardest to utilize the additional datasets provided by organizers at [Human Organ Atlas](https://human-organ-atlas.esrf.eu). 

In the end, I decided to use to following with the help of pseudo-labeling:
- LADAF-2020-31 kidney
- LADAF-2020-27 spleen

In other words, after examining the pseudo annotations on spleen, I realized that they are quite good and should serve as a good regularization method. 

Additionally, I tried to use heart + brain + lung. However, my models make semi-accurate predictions for lung, but horrible for heart + brain. So in the end I decided to stick with kidney + spleen. 

### Pseudo annotations
I would say there are 2 important points

The first one is don't pseudo annotate everything right away. In order to create full pseudo annotations, I run a 4-step process:
- Train on kidney_1. Pseudo-annotate kidney_2
- Train on kidney_1 + kidney_2. Pseudo-annotate the 2020-31 kidney.
- Train on kidney_1 + kidney_2 + 2020-31-kidney. Pseudo-annotate 2020-27 spleen.
- Train on kidney_1 + kidney_2 + 2020-31-kidney + 2020-27 spleen. 

The second point is that don't use hard labels. In other words, don't apply thresholding to the predictions. Simply use soft labels (predictions are sigmoided to be in the range of [0,1]) for training. 

### Loss 

My baseline go-to loss in semantic segmentation is `CE + Dice + Focal`. This worked quite well in this competition. However, since we have a surface metric, I wanted to weight the boundaries of masks more heavily. 

- What didn't work: losses I found in open-source repositories (like Hausdorff Distance loss).
- What worked really well in terms of Surface Dice, FP and FN on validation: CE with x2 weights for boundaries. 

So in the end I decided to use `CE_boundaries + Dice + Focal` for most of my models, and `CE_boundaries + Twersky + Focal` for a single model.

Twersky was focusing more on FN rather than FP, but more on that in the next section. 

```python
class BoundDiceFocalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, bound_alpha=1.0, bound_weight, dice_weight, focal_weight):
        super().__init__()
        self.bound = EdgeEmphasisLoss(alpha=bound_alpha)
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.bound_weight = bound_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds, gt, boundaries):
        return (
            self.bound_weight * self.bound(preds, gt, boundaries)
            + self.dice_weight * self.dice(preds, gt)
            + self.focal_weight * self.focal(preds, gt)
        )

class EdgeEmphasisLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(EdgeEmphasisLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, boundaries):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply the edge weighting
        weighted_loss = bce_loss * (1 + self.alpha * boundaries)

        # Average over the batch
        return weighted_loss.mean()
```

### Preprocessing
After analyzing initial models and its errors, I realized that my biggest issue is FN, not FP. In other words, my models simply don't see some masks, mostly the small ones. 

So I decided to increase the resolution of my trainings with crops from 512x512 to 1024x1024. However, after a couple of hours of training it hit me: that doesn't make much sense. By going from 512x512 to 1024x1024 I don't really increase resolution (each pixels holds the same real-world size), just the context, and 512x512 seemed like a big-enough context already. 

Instead, I decided to do the following: 
```python

class UnetUpscale(nn.Module):
    def __init__(
        self, encoder_name, decoder_use_batchnorm, in_channels, classes, upscale_factor, encoder_weights="imagenet"
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.model = Unet(
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            decoder_use_batchnorm=decoder_use_batchnorm,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, (x.shape[-2] * self.upscale_factor, x.shape[-1] * self.upscale_factor), mode="bilinear"
        )
        x = self.model(x)
        x = torch.nn.functional.interpolate(
            x, (x.shape[-2] // self.upscale_factor, x.shape[-1] // self.upscale_factor), mode="bilinear"
        )
        return x

```

This approached worked really well and I could clearly see improvements both on CV, and LB.

### Models
I used only U-Net models from SMP with different backbones. Tried a lot of things, but for final ensembles decided to settle on the following:
- effnet_v2_s
- effnet_v2_m
- maxvit_base
- dpn68 

Maxvit was trained on 512x512 crops, effent and dpn - on 512x512 with x2 interpolation. Crops were used from xy, xz, and yz axes. During inference, I use the same crops resolution with overlaps of crops_size / 2 (so that's 256). In other words, sliding window approach.

Augmentation were medium-level in terms of intensity. 

```python
return A.Compose(
    [
        A.ShiftScaleRotate(
            p=0.7,
            shift_limit_x=(-0.1, 0.1),
            shift_limit_y=(-0.1, 0.1),
            scale_limit=(-0.25, 0.25),
            rotate_limit=(-25, 25),
            border_mode=cv2.BORDER_CONSTANT,
            # rotate_method="largest_box",
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.25, 0.25),
            p=0.5,
        ),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.OneOf(
            [
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
            ],
            p=0.2,
        ),
        AT.ToTensorV2(),
    ],
    )

```

### Post processing
I tried to use cc3d to remove small objects, it made weak models better, but no difference for ensemble.

### Private resolution
Now, this part is really tricky. My huge thanks to the organizers for announcing the test resolutions. It sincerely warms my heart to see organizers interact with participants that much here on the forum. Really, thank you. 

One approach is not to do anything. You train your model on 50um/voxel, inference on 63um/voxel. Considering I use conv-based backbones (except for maxvit) that have some level of scale-invariance + have scale augs in validation, this might work.

The second approach is to do rescaling. I believe the correct approach for rescaling is the following: 

```python
if test_kidney == 6:
    private_res = 63.08
    public_res = 50.0
            
    scale = private_res / public_res
            
    d_original, h_original, w_original = test_kidney_image.shape
    test_kidney_image = torch.tensor(test_kidney_image).view(1, 1, d_original, h_original, w_original)
    test_kidney_image = test_kidney_image.to(dtype=torch.float32)
    test_kidney_image = torch.nn.functional.interpolate(test_kidney_image, (
        int(d_original*scale),
        int(h_original*scale),
        int(w_original*scale),
    ), mode='trilinear').squeeze().numpy()

```
...
```python

d_preds, h_preds, w_preds = preds_ensemble.shape 
preds_ensemble = preds_ensemble.view(1, 1, d_preds, h_preds, w_preds)
preds_ensemble = preds_ensemble.to(dtype=torch.float32)
            
preds_ensemble = torch.nn.functional.interpolate(preds_ensemble, (
    d_original,
    h_original,
    w_original,
), mode='trilinear').squeeze()
```

So we do 3D resize instead of 2D one: re-scale image from 63um (private) to 50um (public + CV), compute predictions, and re-scale them back to 63um. Simply going for 2D would work as well, but theoretically you end up with different spatial and temporal resolutions in that case. 

This trick helped. To give a single point (I don't have much else): the same ensemble scores 0.634 on private without interpolation, and 0.670 - with interpolation. 

To be honest, I didn't think it would make that much difference. I tried the following experiment locally: 
- Download kidney in 25um resolution. Compute predictions in 25um, interpolate them to 50um, compute metrics. This approach brought my 0.92 surface dice to 0.895. Which is quite good, considering we're talking about x2 interpolation in all 3 directions (that's 8 times less volume) and the fact that it's harder to detect small objects in smaller resolution.
- Download kidney in 25um resolution. Interpolate image to 50um, compute predictions, compute metrics. This approach essentially provided the same metrics as in the case of simply using 50um from organizers. 

So even though I didn't really think interpolation is that important, it also didn't hurt (I was afraid of interpolation artifacts), so I used it for both final subs. 

### Final subs
Both subs have an ensemble of 3 models, each inferenced on all 3 axes without TTA (TTA took too much time, and didn't really help on CV). 
- First sub. CV: 0.84 (kidney_1), Public: 0.768. Private: 0.566
`Maxvit_ce_dice_focal` + `effnet_v2_s_ce_dice_focal` + `effnet_v2_m_ce_dice_focal` trained on kidney_3, validated on kidney_1. This approach didn't work that well on CV, and also on Public and Private. 
- Second sub. CV: 0.923 (kidney_3), Public: 0.855. Private: 0.691
`Maxvit_ce_dice_focal` + `effnet_v2_s_ce_bounds_dice_focal` + `dpn_68_ce_bounds_twersky_focal`.

Code:
- Inference notebook [link](https://www.kaggle.com/code/ivanpan/final-submission/notebook)
- Training code [link](https://github.com/ivanpanshin/segment-vasculature-5th-place)
