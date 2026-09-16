# 4th place: Simple ResNet18 classification

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #4
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583411

Thanks to kaggle and everyone involved for hosting this exciting competition. It was a great learning experience and it was very interesting to see how much of our 1st place cryo ET methodology could be applied here. Thanks to @bloodaxe for this great team experience. I would also like to thank the Armed Forces of Ukraine for providing safety and security for my team mate to participate in this competition. 

## TLDR
The solution is an ensemble of a simple 3D-ResNet18 Classifier and object detection models from MONAI. We also used MONAI for augmentations, and exported models via jit or TensorRT, which gave significant speedup and enabled us to have a slightly larger ensemble. We use the additional data shared by @brendanartley 

This post covers the ResNet18 classification based approach. For object detection part see @bloodaxe writeup: 4th place solution [[Object Detection Part]](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/583228)

## Cross validation
I split the original training data by Voxel Size and the external data by dataset id, to somewhat mimic train/ test difference. 4 Folds were used. Correlation with LB was not very good, so I mainly relied on LB score for feedback.

## Data preprocessing/ augmentations
3D images were scaled to a fixed voxel size of 15.6 and saved to disk using int8.
Since models are trained from scratch, augmentations were essential to prevent overfitting.
I used RandomCrop (size 96x160x160), Flip on each axis within the torch dataloader, and additionally scale + rotation on GPU (all from MONAI). Additionally, I used a customized implementation of MixUp which was highly effective to train longer and prevent overfitting. I implemented a version which makes sure to not have more than 1 motor in a mixed patch. Additionally, positive samples, i.e. crops with motor in it,  were oversampled by having a total fraction of 12.5%

## Model
Modelling was quite interesting in this competition. I started with the 3D UNET from our 1st place solution of CryoET competition, which worked already quite well. After learning about the forgiveness of the competition metric with respect to localization, I tried to simplify the model further and get rid of any decoder altogether, since the 32x downscaled model output should already be enough. Surprisingly, a simple ResNet3D encoder worked. My approach works the following:

Input for the classification model are 96x160x160 image patches, which resulted in a feature map of 512x3x5x5 leaving the resnet backbone. I flatten the 3x5x5 output "pixels" and use a simple fully conected 512->1  layer to have a binary prediction for each pixel, which are basically 75 classes, determin the motor location. I also add an additional class to reflect having no-motor in the crop. So in total its a simple 3D-ResNet18 classifier with 76 classes, which is trained with CrossEntropy Loss. 
For inference I used a sliding window aproach with an overlap of 0.5. For motor localisation, simplytake  the patch with max prediction value of the 75 classes. Then use the patch location + and offset coming from the 3x5x5 grid to determine the final localisation. This very simple and fast model scores 0.875 on public LB (5th place)  individually! 



The architecture might be easier to understand via code:


```python
import monai.networks.nets as mnn

def downscale(y, scale=32):
    bs, c, d, h, w = y.shape
    idxs = torch.where(y>0)
    for item in idxs[2:]:
        item //= scale
    y2 = torch.zeros((bs,c,d//scale,h//scale,w//scale), dtype=y.dtype, layout=y.layout, device=y.device)
    y2[idxs] += 1
    return y2

cfg.backbone_args = dict(model_name='resnet18',
                         spatial_dims=3,    
                         pretrained=False, 
in_channels=1)

class Net(nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.backbone = mnn.ResNetFeatures(**cfg.backbone_args)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.reg_head = nn.Conv3d(512,1,kernel_size=1,stride=1)
        self.cls_head = torch.nn.Linear(512,1)        
           
    def forward(self, batch):

        x = batch['input']
        
        out = self.backbone(x)[-1]
        loc_logits = self.reg_head(out)
        cls_logits = self.cls_head(self.global_pool(out).flatten(1))

        loss = self.custom_loss(y,loc_logits,cls_logits)
        outputs = {'loss':loss,'logits':loc_logits}
        return outputs

    def custom_loss(self, target, logits, cls_logits):

        y2 = downscale(target,scale=32)
        l = logits.flatten(1)
        y3 = y2.flatten(1)
        y3 = torch.cat([y3,1-y3.max(1)[0][:,None]],dim=-1)
        l2 = torch.cat([l,cls_logits],dim=-1)
        l_cls = DenseCrossEntropy1D()(l2,y3)
        return l_cls

```

For threshoolding I used a quantile based a aproach as this was much more stable when comparing different models. 

The model was trained with bf16, and each fold needs about 17h on a single A100 for training. Although the model is small and simple, I tried a lot of other architectures, and alternatives performed much worse. So I sticked with this one. 

Code base is very close to[ 1st place solution of cryo et ](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation), so I will save me the trouble to publish this one. 
Cheers. Questions welcome.
