# 2nd Place Solution [Save the Prostate]

Competition: prostate-cancer-grade-assessment
Rank: #2
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169108

First of all, Thank you very much to organizers.
Second I would like to thanks my team. We had such a positive, encouraging working environment. Our team contribution generates most of the ideas which you will read below and are shared by members. 


#Simple Resnet34 (DrHB)

# Image Preprocessing 
I used medium resolution, the only preprocessing I did was to remove the white background and store medium resolution on SSD drive: 

```
#function taken from R Guo
def crop_white(image, value: int = 255):
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) &lt; value).nonzero()
    xs, = (image.min(0).min(1) &lt; value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

```

# Cleaning data
Like in APTOS competition, it was essential to clean images from pen marks, etc. I have used excellent work from this post: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/151323 This also reduced the gap between CV and LB


# Image Augmenatiosn
Augmentation occurred at two levels.  (Slide and Tile): 
### 1) Full slide
After the biopsy slide is open, we do random padding and applying one of the following transformations (similar to R Guo). 

```
def get_transforms_train():
    transforms=A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5,value=(255,255,255)),
            A.OneOf([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], p=0.3
            )
        ]
    )
    return transforms
```

### 2) Tile
For each tile I used standard fastai `GPU` augmentations: `rotate=(-10, 10`),` flip vertically (p=0.5)`. For the padding I used `reflection`  it gave a slight boost on CV 

# Model 

I decided to use a very simple model resnet34 but throughout competitions ended up doing few modifications 

### 1) Making square features:
The main idea is builds up on @iafoss. Aftter resnet enccoder we reshape features to look like a square in a following way: `x = x.view(x.shape[0], x.shape[1], x.shape[2]//int(np.sqrt(N)), -1)` . Here `N` represents Number of Tiles.  After this we pass all the features to `SqueezeExcite` Block

#### 2) SqueezeExcite block
After reshaping features, we added 1 SE block to enable the network to learn features for individual slides based on tiles. 

experiment done by @cateek 
```
#code adopted 
#https://github.com/rwightman/pytorch-image-models/tree/master/timm/models
def make_divisible(v, divisor=8, min_value=None):
   min_value = min_value or divisor
   new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
   # Make sure that round down does not go down by more than 10%.
   if new_v &lt; 0.9 * v:
      new_v += divisor
   return new_v
def sigmoid(x, inplace: bool = False):
   return x.sigmoid_() if inplace else x.sigmoid()
class SqueezeExcite(nn.Module):
   def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
             act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
      super(SqueezeExcite, self).__init__()
      self.gate_fn = gate_fn
      reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
      self.act1 = act_layer(inplace=True)
      self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
   def forward(self, x):
      x_se = self.avg_pool(x)
      x_se = self.conv_reduce(x_se)
      x_se = self.act1(x_se)
      x_se = self.conv_expand(x_se)
      x = x * self.gate_fn(x_se)
      return x
```

### 3) Pooling Layer 
Once the feature passed thru SqueezeExcite Layer, I did Normal pooling. Our experiment showed that the batch normalization layer was messing with the last layer's features, so we removed it and saw a slight jump on local cv. 



```
self.pool = nn.Sequential(AdaptiveConcatPool2d(),
                          Flatten(),
                          nn.Linear(2*nc,512),
                          nn.ReLU(inplace=True),
                          nn.Dropout(0.4),
                          nn.Linear(512,7), 
```


### 4) Final Head
I used two heads. One head was for classification second was for regression. I noticed that training with two looses makes training much smoother (with sigmoid trick below) and yields higher local CV (0.88 -&gt; 0.90). In the final prediction, I use output only for the regression head. 

One small modification that I did before calculating loss is that the regression head used sigmoid to scale outputs between (-1. 6.). This enables much smoother training without bumps and faster convergence.


```
#idea taken from fastai
def sigmoid_range(x, low, high):
    return torch.sigmoid(x) * (high - low) + low 
```

# Training 
I trained in two phases. In the First phase was trained with 49 tiles and later finetuned with 81 tiles. Both phases were using standard one cycle.

# Final Model 
I trained 5 fold wich resulted on the CV of 0.911 and PB: 0.922.

Our Best Ensemble was simple average. Of 4 models. 

```
@drhb resnet34 5 FOLD (CV -0.911) + 
@rguo97  5 FOLD (two stage attention model CV 0.92 ) + 
@xiejialun  FOLD (EFNET) (CV 0.915-0.917) +
@cateek  Se 1 FOLD (CV -0.91) Final Standing
```

@xiejialun https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169303
@rguo97 https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169108#940504

 `LB: 0.914 PB: 0.937`
