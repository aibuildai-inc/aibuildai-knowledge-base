# 4th place solution

Competition: severstal-steel-defect-detection
Rank: #4
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114716

Hi everyone, and congratulation to all the participants!

We all entered this competition a little less than a month ago, right after the end of the APTOS competition.
We were solving it separately and merged only about a day before the Merger deadline because of the emptying feeling of despair.
It was the first competition, where the 'fit_predict' could not get into the bronze zone. Besides, heavy encoders were virtually as successful as simple resnet34.
At first, here are the things that we tried, and that did not work at all:

**I tried:**
 - Adding hard negative mining by resampling dataset every epoch. Weights were chosen as the inverse of the Dice score per image. Absolutely no difference;
 - Adding ArcFace to embeddings. It should help to distinguish classes better. One more nope;
 - Training multi-stage network (inspired by pose detectors). It should help mimic bad markings. Performed worse than single-stage;
 - Adding label smoothing. Nope again.

@bloodaxe **tried:**
 - Adding mixup and Poisson blending to increase the number of images with defects;
 - Trainig on double-sized crops: 256x1600 -&gt; crop(256x512) -&gt; resize(512x1024). Nope;
 - Adding result of anisotropic segmentation to the input of the neural network, no gain;
 - Training HRNetV2 with full resolution. 22 hours with 8xV100 and no better than ResNet34.

**Now, things that actually worked:**
 - Training two-headed NN for segmentation and classification.
 - Combine heads at inference time as with soft gating (mask.sigmoid() * classifier.sigmoid())
 - Focal loss / BCE + Focal loss
 - Training with grayscale instead of gray-RGB
 - FP16 with usage of Catalyst and Apex

I trained only single-fold models and @bloodaxe trained 5-fold CV.

Our individual solutions were no more than at the end of the silver zone of the Public LB, but then we teamed up... and got a bit higher in the silver zone!

Our best (and final) ensemble consisted of 9 models with densenet201, efficientnetb5, resnet34, seresnext50 encoders, some of them with FPN decoders, and some with UNet.
We added 3-flip TTA and averaged logits of the models, and soft gating applied. We binarized masks with a 0.55 threshold and zeroed out masks less than 256 pixels.
Our total runtime for private+public is ~30 min.

**Hardware**
We used servers from FastGPU.net with 8xV100 and 4xV100, that greatly reduced our experiment cycle length.

**Some speculation about our rise on private LB:**
 - We did not overfit to the pubic LB;
 - We chose our models by the Dice score without empty masks. We relied on our classifiers and soft-gating for empty masks.
 - We had different seeds/folds/models. Out private scores before merge are not that great (0.89-0.90). But they are much stronger when combined.

**Some moral by** @bloodaxe
 - Keep going. Even if it seems that everything is lost and there is no hope, you still can push a little bit forward and learn something useful along the way;
 - Team up! Exchanging ideas is really beneficial!
