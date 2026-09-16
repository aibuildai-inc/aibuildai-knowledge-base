# 1st Place Solution (code updated)

Competition: blood-vessel-segmentation
Rank: #1
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475522

First of all, we would like to thank Kaggle and the organizers for hosting such a great competition. And also thanks to @hengck23 for the amazing posts, @junkoda for the metric implementation and all other participants for sharing their experiments.
# Overview
Our final submission is an ensemble of two 2.5d convnext tiny unet with 3 channels, and the only differences between these two models are augmentation and number of epochs. Actually the best scored submission is not the selected ensemble but one single model of the ensemble which is 0.835 on private lb.

# Data Preparation
We used all training data **including** kidney_1_voi.
- Multiview slice (x, y, z)
- Normalization: No normalization, just `image = image / 65535.0`
- Whole slice instead of tiles and all slices resized or cropped to 1536x1536. 
- Augmentations:
```
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.Affine(scale={"x":(0.7, 1.3), "y":(0.7, 1.3)}, translate_percent={"x":(0, 0.1), "y":(0, 0.1)}, rotate=(-30, 30), shear=(-20, 20), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
    ], p=1.0),
    A.OneOf([
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5)
    ], p=0.4),
    A.OneOf([
        A.Resize(1536, 1536, cv2.INTER_LINEAR, p=1),
        A.Compose([
            RandomResize(1536, 1536, scale_limit_x=0.5, scale_limit_y=0.5, p=1),
            A.PadIfNeeded(1536, 1536, position="random", border_mode=cv2.BORDER_REPLICATE, p=1.0),
            A.RandomCrop(1536, 1536, p=1.0)
        ], p=1.0),
    ], p=1.0),
    A.GaussNoise(var_limit=0.05, p=0.2),
])
```
- Random 3D rotation to get slices that is not necessarily parallel to axes. The best scored submission used random 3d rotation for augmentation and trained for more epochs.


# Modeling & Training
- We used unet from SMP with convnext tiny backbone, replaced BatchNorm and ReLU to GroupNorm and GELU and added an extra convolution stem. The input size for all models is 3x1536x1536.
```
self.extra_stem = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, 2, 1),
    LayerNorm2d(out_channels),
)
```
- For loss function, we used 1.0 focal loss, 1.0 dice loss, 0.01 [boundary loss](https://arxiv.org/abs/1812.07032) and 1.0 custom loss. The custom loss is inspired by @hengck23's [post](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/456118#2583472) and @junkoda's [metric implementation](https://www.kaggle.com/code/junkoda/fast-surface-dice-computation).
```
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        power = 2**np.arange(0, 8).reshape(1, 1, 2, 2, 2).astype(np.float32)
        area = create_table_neighbour_code_to_surface_area((1, 1, 1)).astype(np.float32)
        self.power = nn.Parameter(torch.from_numpy(power), requires_grad=False)
        self.kernel = nn.Parameter(torch.ones(1, 1, 2, 2, 2), requires_grad=False)
        self.area = nn.Parameter(torch.from_numpy(area), requires_grad=False)
        
    def forward(self, preds, targets):
        """
        preds: tensor of shape [bs, 1, d, h, w]
        targets: tensor of shape [bs, 1, d, h, w]
        """
        bsz = preds.shape[0]

        # voxel logits to cube logits
        foreground_probs = F.conv3d(F.logsigmoid(preds), self.kernel).exp().flatten(1)
        background_probs = F.conv3d(F.logsigmoid(-preds), self.kernel).exp().flatten(1)
        surface_probs = 1 - foreground_probs - background_probs

        # ground truth
        with torch.no_grad():
            cubes_byte = F.conv3d(targets, self.power).to(torch.int32)
            gt_area = self.area[cubes_byte.reshape(-1)].reshape(bsz, -1)
            gt_foreground = (cubes_byte == 255).to(torch.float32).reshape(bsz, -1)
            gt_background = (cubes_byte == 0).to(torch.float32).reshape(bsz, -1)
            gt_surface = (gt_area > 0).to(torch.float32).reshape(bsz, -1)
        
        # dice
        foreground_dice = 2 * (foreground_probs*gt_foreground).sum(-1) / (foreground_probs.sum(-1)+gt_foreground.sum(-1)).clamp(1e-6)
        background_dice = 2 * (background_probs*gt_background).sum(-1) / (background_probs.sum(-1)+gt_background.sum(-1)).clamp(1e-6)
        surface_dice = 2 * (surface_probs*gt_area).sum(-1) / ((surface_probs+gt_surface)*gt_area).sum(-1).clamp(1e-6)
        dice = (foreground_dice + background_dice + surface_dice) / 3
        return 1 - dice.mean()
```
- For optimization, we used AdamW and CosineAnnealingLR from 1e-4 to 0 with warmup. All models were trained for 20 epochs with a batch size of 8 and 4 gradient accumulation steps, except for the model with 3d slice rotation augmentation which was trained for 30 epochs.
# Inference
- Inference on 3 axes with 8xTTA.
- We tried different resize methods for inference. For the best scored submission, all slices are simply resized to 3072x3072; for the selected submission, we used a dynamic scale factor that `(h*scale)*(w*scale)=3200*3200`.
- The threshold used for submission is 0.4, and the optimal threshold based on cv and lb is about 0.4~0.5.
- `torch.compile()` gave about 2x acceleration so that we were able to inference with high resolution and TTAs.
# What didn't work
- 3d models.
- External data and pseudo labels.
- Transformers.
- Stacking more slices (>3) for 2.5d model.
# Results
|  | **Model** | **Slice Rotation** | **Inference size** | **Public Score**| **Private Score**|
| --- | --- | --- | --- | --- | --- |
| 1 | convnext_tiny |  | 3072 | 0.889 | 0.682 |
| 2 | convnext_tiny | ✓ | 3072| 0.888 | 0.830 |
| 3 | convnext_tiny | ✓ | 3072| 0.867 | **0.835** |
| 4 | ensemble(1+2) | - | 3200 | **0.898** | 0.744(selected) |
| 5 | ensemble(1+2) | - | 3200(dynamic) | 0.895 | 0.774(selected) |
# Links
- [training code](https://github.com/jing1tian/blood-vessel-segmentation)
- inference code
    - [final submission ensemble 0.774106](https://www.kaggle.com/code/clevert/sennet-1st-place-solution)
    - [3d rotate single model 0.835346](https://www.kaggle.com/code/clevert/sennet-unet-convnext-3d-rotation)
