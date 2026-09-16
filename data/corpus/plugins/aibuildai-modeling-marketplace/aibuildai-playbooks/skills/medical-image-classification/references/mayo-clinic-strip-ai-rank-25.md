# 25th Solution - Custom MIL with timm backbones

Competition: mayo-clinic-strip-ai
Rank: #25
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/357898

When I first started this competition 2 weeks ago, I got RSNA-MICCAI Brain Tumor Radiogenomic Classification flashbacks. These two competitions are quite similar in terms of low signal but this one was slightly better and less random. I had some decent experiences in that competition and I applied those things here.

Inference Notebook: https://www.kaggle.com/code/gunesevitan/mayo-clinic-strip-ai-inference
GitHub Repository: https://github.com/gunesevitan/mayo-clinic-strip-ai

## Dataset

I created a pre-computed dataset of instances for training more efficiently. Pre-computed dataset consist of 16 instances of size 1024x1024 with 3 channels (RGB) per image. Dataset creation process can be simplified to

* Compress image with JPEG compression (100% JPEG quality)
* Resize longest edge to 20.000 pixels
* Extract non-overlapping instances and pad them with white background
* Sort instances by their sums in descending order and take top 16 instances

## Models

I used Multiple Instance Learning (MIL) models for processing instances. My MIL model is slightly different than the one shared in this competition. I think the model in [this](https://www.kaggle.com/code/analokamus/a-sample-of-multi-instance-learning-model) notebook concatenates feature maps on height dimension which felt weird to me, so I tried different pooling and concatenation methods. 

```
class ConvolutionalMultiInstanceLearningModel(nn.Module):

    def __init__(self, n_instances, model_name, pretrained, freeze_parameters, aggregation, head_class, head_args):

        super(ConvolutionalMultiInstanceLearningModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=head_args['n_classes']
        )

        if freeze_parameters is not None:
            # Freeze all parameters in backbone
            if freeze_parameters == 'all':
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            else:
                # Freeze specified parameters in backbone
                for group in freeze_parameters:
                    if isinstance(self.backbone, timm.models.DenseNet):
                        for parameter in self.backbone.features[group].parameters():
                            parameter.requires_grad = False
                    elif isinstance(self.backbone, timm.models.EfficientNet):
                        for parameter in self.backbone.blocks[group].parameters():
                            parameter.requires_grad = False

        self.aggregation = aggregation
        n_classifier_features = self.backbone.get_classifier().in_features
        input_features = (n_classifier_features * n_instances) if self.aggregation == 'concat' else n_classifier_features
        self.classification_head = eval(head_class)(input_features=input_features, **head_args)

    def forward(self, x):

        # Stack instances on batch dimension before passing input to feature extractor
        input_batch_size, input_instance, input_channel, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_instance, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)
        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.aggregation == 'avg':
            # Average feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.mean(x, dim=1)
        elif self.aggregation == 'max':
            # Max feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.max(x, dim=1)[0]
        elif self.aggregation == 'logsumexp':
            # LogSumExp feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.logsumexp(x, dim=1)
        elif self.aggregation == 'concat':
            # Stack feature maps on channel dimension
            x = x.contiguous().view(input_batch_size, input_instance * feature_channel, feature_height, feature_width)

        output = self.classification_head(x)
        return output

    def __init__(self, n_instances, model_name, pretrained, freeze_parameters, aggregation, head_class, head_args):

        super(ConvolutionalMultiInstanceLearningModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=head_args['n_classes']
        )

        if freeze_parameters is not None:
            # Freeze all parameters in backbone
            if freeze_parameters == 'all':
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            else:
                # Freeze specified parameters in backbone
                for group in freeze_parameters:
                    if isinstance(self.backbone, timm.models.DenseNet):
                        for parameter in self.backbone.features[group].parameters():
                            parameter.requires_grad = False
                    elif isinstance(self.backbone, timm.models.EfficientNet):
                        for parameter in self.backbone.blocks[group].parameters():
                            parameter.requires_grad = False

        self.aggregation = aggregation
        n_classifier_features = self.backbone.get_classifier().in_features
        input_features = (n_classifier_features * n_instances) if self.aggregation == 'concat' else n_classifier_features
        self.classification_head = eval(head_class)(input_features=input_features, **head_args)

    def forward(self, x):

        # Stack instances on batch dimension before passing input to feature extractor
        input_batch_size, input_instance, input_channel, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_instance, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)
        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.aggregation == 'avg':
            # Average feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.mean(x, dim=1)
        elif self.aggregation == 'max':
            # Max feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.max(x, dim=1)[0]
        elif self.aggregation == 'logsumexp':
            # LogSumExp feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.logsumexp(x, dim=1)
        elif self.aggregation == 'concat':
            # Stack feature maps on channel dimension
            x = x.contiguous().view(input_batch_size, input_instance * feature_channel, feature_height, feature_width)

        output = self.classification_head(x)
        return output
```

Aggregation parameter tunes the MIL pooling or aggregating part of the model.
* `avg`: average all feature maps on instance dimension
* `max`: max all feature maps on instance dimension (this was mentioned on many papers)
* `logsumexp`: logsumexmp all feature maps on instance (I saw this on one paper so I added it as well)
* `concat`: concatenate all feature maps on channel dimension

`concat` worked best for me even though the feature map counts multiplied with number of instances. I was working on an attention mechanism for instance feature maps but I didn't have enough time and I eventually ditched the idea.

## Validation

5 folds of cross-validation is used as the validation scheme. Splits are stratified on `binary_encoded_label`, `patient_id` doesn't overlap between folds, and dataset is shuffled before splitting. Best random seed is selected by minimizing the standard deviation of `binary_encoded_label` mean between folds which ensures target means of folds are close to each other.

## Training

Loss Function: Negative log likelihood loss is used as the loss function. Binary labels are converted to multi-class labels by simply doing
```

inputs = torch.sigmoid(inputs)
inputs = torch.cat(((1 - inputs), inputs), dim=-1)
```

Optimizer: AdamW with 1e-5 initial learning rate and learning rate is cycled to 1e-6 back-and-forth after every 300 steps.

Batch Size: Training: 4 - Validation: 8

Early Stopping: Early Stopping is triggered after 5 epochs with no validation loss improvement

## Training Augmentations

Training augmentations are applied to all instances randomly.

* Resize 256x256 or 224x224
* Luminosity standardization by 10% chance
* Horizontal flip by 50% chance
* Vertical flip by 50% chance
* Random 90-degree rotation by 25% chance
* Random hue-saturation-value by 25% chance
* Coarse dropout or pixel dropout by 10% chance
* Normalize by instance dataset statistics

## Ensemble

8 models are used in the ensemble. Those models are selected based on their ROC AUC scores. Models with OOF ROC AUC score greater than 0.65 are qualified for the ensemble. My best single model was DenseNetBlur121d which is quite interesting and worth more exploring.

* DenseNet121 (ImageNet pre-trained weights) - 256x256 16 instances
* DenseNetBlur121d (ImageNet pre-trained weights) - 256x256 16 instances
* DenseNet169 (ImageNet pre-trained weights) - 256x256 16 instances
* EfficientNetB2 (ImageNet pre-trained weights) - 256x256 16 instances
* EfficientNetV2 Tiny (ImageNet pre-trained weights) - 256x256 16 instances
* CoaT Lite Mini (ImageNet pre-trained weights) - 224x224 16 instances
* PoolFormer 24 (ImageNet pre-trained weights) - 224x224 16 instances
* Swin Transformer Tiny Patch 4 Window 7 (ImageNet pre-trained weights) - 224x224 16 instances

Blending is utilized on selected models. Blend weights are found by trial and error.

```
blend_weights = {
    'mil_densenet121_16_256': 0.10,
    'mil_densenet169_16_256': 0.10,
    'mil_densenetblur121d_16_256': 0.25,
    'mil_efficientnetb2_16_256': 0.125,
    'mil_efficientnetv2rwt_16_256': 0.125,
    'mil_coatlitemini_16_224': 0.10,
    'mil_poolformer24_16_224': 0.10,
    'mil_swintinypatch4window7_16_224': 0.10
}

```
Linear stacking is also utilized on selected models by fitting a linear regression model on training predictions and targets.

## Post-processing

Predictions are averaged among patient_id groups and 0.21 is added to aggregated predictions for making the prediction 0.5 centered. That constant is selected by maximizing the OOF score.

## Results

Quantitative results are provided as cross-validation, public and private leaderboard scores.
Best single model and ensemble cross-validation scores are highlighted.

|                                             | CV ROC AUC | CV Weighted Log Loss | Public Leaderboard | Private Leaderboard |
|---------------------------------------------|------------|----------------------|--------------------|---------------------|
| DenseNet121                                 | 0.6501     |                      |                    |                     |
| DenseNetBlur121d                            | **0.6628** |                      |                    |                     |
| DenseNet169                                 | 0.6551     |                      |                    |                     |
| EfficientNetB2                              | 0.6501     |                      |                    |                     |
| EfficientNetV2 Tiny                         | 0.6500     |                      |                    |                     |
| CoaT Lite Mini                              | 0.6503     |                      |                    |                     |
| PoolFormer 24                               | 0.6536     |                      |                    |                     |
| Swin Transformer Tiny Patch 4 Window 7      | 0.6497     |                      |                    |                     |
| Blend + patient_id Aggregation + Adjustment | 0.6822     | 0.6471               | 0.6                | **0.67897**                    |
| Stack + patient_id Aggregation + Adjustment | **0.6842** | **0.6387**           | **0.6**            | 0.68666                    |

I started overfitting when I added more and more models to ensemble. My stack was also overfitting a bit but my best submission is my blend.

## What didn't work

* Decaying learning rate
* Freezing backbone parameters
* Average and max MIL pooling
* Lots of other timm models
* Binary cross-entropy loss, focal loss, weighted log loss
* Label smoothing
* Different number of instances with larger sizes
