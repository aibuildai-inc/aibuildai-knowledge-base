# Public/Private 3rd Solution

Competition: hotel-id-to-combat-human-trafficking-2022-fgvc9
Rank: #3
Source: https://www.kaggle.com/c/hotel-id-to-combat-human-trafficking-2022-fgvc9/discussion/328237

Thank you for host,competitors,team member @ks2019 
It is a great competiton!
here is our team summary.

# Summary
- ArcFaceSubcenter + Dynamic Margin
- Many Backbones(Swin,ConvNeXt, ResNet200D, EfficientNet etc..)
- use External Data(FGVC8)
- use Logits

## Modeling
|Model|Data|ImageSize|Public|Private|
| --- | --- |
|Swin Large|Comp|512|0.587|0.584|
|ConvNeXt XLarge|Comp + External|384|0.662|0.656|
|Swin Large|Comp+ External|384|0.659|0.657|
|EfficientNetB7 + DOLG|Comp + External|448(stride1)|0.644|0.639|
|ConvNeXt XLarge|Comp + External|512|0.677|0.672|
|ResNet200D + DOLG|Comp + External|640|0.651|0.654|
|EfficientNetB6 + DOLG|Comp + External|640|0.635|0.642|
|EfficientNetV2S + DOLG|Comp + External|1024|0.645|0.642|
|EfficientNetV2M + DOLG|Comp + External|896|0.666|0.666|

## Pseudo Labeling
We use FGVC9 and FGVC8 competiton data but FGVC8 don't have competiton labels.
We annotated label to these data using Pseudo Labeling.

First, We used KNN matching FGVC9 training and FGVC8 dataset.
also if threshold < 0.5, these data used training.

Next, We adjust labeling. FGVC8 have same hotel list.
I use mean aggregation method

## Prediction
I use logits because test have a mask, but train do not have a mask.
I think it is difficult to match training and test so I decide to compare knn vs logits
Logits is better than knn in my experiments.

My inference time is about 2hours.

## did not work
- used Hotel50K
- gradient checkpointing
