# Team FAMAS. 5th place solution.

Competition: waveform-inversion
Rank: #5
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587443

Thanks to kaggle and everyone involved for hosting such an intense competition. It was a great 3-week run for our team FAMAS — @forcewithme, @alehandreus, @kevin1742064161, @arsenypoyda, and @samson8. The solution we achieved would not have been nearly as strong without the contributions of every team member.

# Brief summary

Our solution is a hill-climb ensemble of 6 models trained on additionaly augmented data and post-processed family-wise using a differentiable version of a data simulator.

# Detailed summary

## 1. Resources.

In total, we used 3x8 A100 + 2x4 A100 + 8xL20 to run our experiments, train models and apply various optimization techniques. Moreover, we rented cloud 32x4090 and 4x5070, which enabled us to further optimize our results.

## 2. Data augmentations / additional data generation.

The 'seis-images' are the solutions of the wave equation `p(r,t)` with the medium given by the 'vel-images' `v(r)`. Therefore, we can extend the training data by solving a forward problem: generating seis-images based on vel-images. @arsenypoyda successfully managed to restore host's simulator which, which made this data generation process possible. That allowed us to use complex vel-image transformations, e.g. CutMix:



We have also created MixUp, both vertical and horizontal stretches, wave deformations, smooth multiplications and artificial faults. Basically, each unique combination could be counted as a new family. Moreover, we believe that some families, e.g. CFB are already CutMix'ed versions of some 'easier' families.

There is a problem though. Generating one additional sample takes approximately 0.4s (even with CUDA acceleration). That's why the proposed approach is only useful for creating additional train data.

Of course, there are ways to apply some augmentations 'online' also. The most trivial augmentation is the flip augmentation that was used in most of the public notebooks. Additionally, if `p(r,t)` is described by `v(r)`, then `p(r,a⋅t)` is described by `a⋅v(r)`. Physically, this means that if we multiply the vel-image by `a`, the time axis of the corresponding seis-image will be compressed by `a`, or, in other words, if the velocity is higher, the propagation time is lower. Strictly speaking, this reasoning is only valid for a delta-impulse signal source. However, even when exposed to a long-duration source (as in this competition), we were able to slightly improve the model. This is effectively a way to implement runtime augmentations. We haven't used it in our final pipeline, but we still consider it as an interesting finding. 

Also, 5 days before the competition end deadline we found out, that SA looks like perlin noise. So we created blob-like structures using Perlin noise, applying Gaussian smoothing, then adding linear gradients and normalizing the values to the required velocity range (1500-4500 m/s):



Basically, we could have generated as much additional SA data as we wanted to improve the validation score on that specific family. Additionally, we found how to improve SA even further (see the post-processing section).

## 3. Model architecture and training.

1\. **Model architecture**: All the models we used can be considered as `caformer/convformer + unet_{pubic/modified}` in structure. Before teaming up and implementing data augmentation, we all observed that increasing the resolution of feature maps could significantly improve CV/LB performance. We found that Convformer showed better performance than Caformer in both CV and LB (-1.0). Our final 6 models include: 3 Convformers with 144 resolution, Caformers with 144, 160, and 256 resolutions. Among these, the  144-resolution performed the best.

The following is the structure diagram of the model:



2\. **Speed up training**: Directly training with a larger resolution is a option—for example, caformer256 follows this approach. However, a more effective strategy is to first train with a smaller resolution and then scale up. For instance, we initially trained convformer72 and later fine-tuned convformer144 by resuming the weights from convformer72 before we figure out how to make data augs.
3\. **Data Generation**: We generated a dataset that is 15 times the size of the official dataset, which can be roughly divided into 8 versions, utilizing all the data augmentation mentioned above. Among these, the more challenging classes have higher weights, with CFB, CVB, FFB, SA, and SB having significantly higher weights than the remaining classes. In the last stages of training, CFB even accounted for more than half of the training set. It is important to note that no matter how is synthetic dataset, it must be combined with the competition dataset for training; otherwise, the performance on the simpler classes will significantly deteriorate.
4\. **Resume training**: We did not attempt to train the model with the entire 15x-sized dataset at once. The first model was trained sequentially on the first 7 versions of the synthetic dataset (e.g., augv1 → augv2 → ... → augv7), ultimately achieving a CV-score of 10.26.The remaining models were trained simultaneously on multiple synthetic datasets in stages, (such as:augv1v2 → augv3v4v5 → augv6v7 → augv8).
5\. **Lower the LR gradually**. The LR strategy for all of the models are close. Before the models haven't break a 13-CV score, we use a LR of 1e-4. After that, if we observed the CV score decreases slowly, we switch to a new version of dataset (e.g. augv6->v7) and lower the LR (1e-4 -> 5e-5 -> 2e-5 -> 1e-5 -> 5e-6 -> 1e-6).

The following table contains information about our models in the final ensemble.

| model | backbone | decoder | CV-Score|
| --- | --- |--- |--- |
| augv7_conv144 | Convformer | modified Unet|10.26 |
| augv8_conv144 | Convformer | modified Unet|10.28 |
| augv567_conv144 | Convformer |modified Unet|10.53 |
| augv8_ca144 | Caformer |modified Unet|11.3 |
| augv8_ca256 | Caformer |public Unet|11.9 |
| augv8_ca160 | Caformer |public Unet|12.0 |

## 4. Post-processing.

4.1. **Ensemble**: We used the hill‐climb method to ensemble our models. Hill‐climb method is an optimization method that, at each step, selects the local change yielding the largest immediate performance gain and repeats until no significant improvement remains. Compared with mean and median methods, hill-climb improves cv by about 0.3-0.4. However, the median method of CFB is 0.4 higher than hill-climb.

4.2. @alehandreus found a way to modify ours vel-to-seis forward simulator in PyTorch as a differentiable function. We used it to refine the predictions of our ensemble:

1. Given `seis_true`, the ensemble outputs `vel_pred`;
2. Differentiable forward simulation on `vel_pred` transforms it into `seis_pred`;
3. `vel_pred` pixels are optimized with gradient descent to minimize the difference between `seis_pred` and `seis_true`.

#### Style A/B families
On these families the method works without any modifications. Still, there were a couple of modifications to improve the result:
1. Set individual LR for each pixel proportional to `vel_pred` values. Larger values usually have more MAE and thus receive larger LR.
2. `seis_pred` and `seis_true` have 1000 time steps. To prevent vanishing gradients at time steps close to 1000, we multipled seis by `linspace` from 0.1 to 10.

#### "Discrete" families
 Firstly, most of the mistakes in our model predictions come from the boundaries/edges between layers:

.jpg?generation=1751345175204562&alt=media)

Because of the strict layer structure, these samples required more sophisticated modifications:

1. Apply Total Variation loss on `vel_pred`. A crucial thing we discovered yesterday is to use p < 1 for the p-norm to encourage sharp edges.
2. We replaced the value of the LR with the maximum value of it and its neighbours. Thus pixels one unit away from the edges still recieved larger LR.

#### FlatVel_B
Here we used a completely different approach. Predicted stripes are nearly perfect and require only slight color tuning. We replaced each stripe with one color and optimized only this one color.

#### Results

| Family    | Estimated improvement | Required iterations |
| -------- | ------- | ------- |
| CurveFault_A  | -    | -    |
| CurveFault_B | 1.5     | 250  |
| CurveVel_A    | 0.4    | 300 |
| CurveVel_B    | 2.0    | 600 |
| FlatFault_A    | -   | - |
| FlatFault_B    | 1.0   | 300 |
| FlatVel_A    |  - | - |
| FlatVel_B    | 0.95 | 80 |
| Style_A    | 14 | 5000 |
| Style_B    | 6 | 5000 |

#### Classifier

To split the test set into families we trained a model with the classifier head. The following are the results on the validation data subset:



We used that classifier to differ Style families from "Discrete" families in the test data, since these family types requires different optimization strategies. Moreover, we used it to perform a family-wise hill-climb ensembling.

Thanks for reading. Questions welcome.

Code: (in progress)
