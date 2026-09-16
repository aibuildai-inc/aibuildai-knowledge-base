# 14th place solution

Competition: quickdraw-doodle-recognition
Rank: #14
Source: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73761

Because I joined this competition with just two weeks left,  I decided to train only basic classifier.
So,  It was really lucky to have achieved a relatively high score.

After reading discussions (especially [Heng](https://www.kaggle.com/hengck23)'s posts), I decided to use following settings.

## Dataset
- *Train*: All images including unrecognized
- *Validation*: 500 images per class
- 3 channel, each are 1/3, 2/3, 3/3 of the total strokes.
- 224 x 224

## Training
- cross entropy loss
- batch size: 256 for se_resnext50, 128 for se_resnext101 and xception.
- adam optimizer
- learning rate 0.00025
- reduce learning rate when MAP@3 has stopped improving by half
- no augmentation

## Inference
- average last ten weights (I saved checkpoints every 5000 step)
- horizontal flip tta with weight 0.5

In total I trained 3 models:

- se_resnext50, se_resnext101, xception

All of them have similar Public LB scores, 0.947x.
After ensembling all of them Public LB 0.950x.

Congratulations to the winners and thanks for all the participants!
