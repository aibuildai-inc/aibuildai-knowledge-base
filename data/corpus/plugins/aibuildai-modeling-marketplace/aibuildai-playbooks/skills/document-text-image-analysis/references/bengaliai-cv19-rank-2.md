# 2nd place solution

Competition: bengaliai-cv19
Rank: #2
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/135966

Thanks a lot to the host and Kaggle for this very interesting competition. We learned a lot and specifically the issue of predicting unseen graphemes in test data was very intriguing and has led to some very elegant and nice solutions as imminent from all the other solution posts. As always, this has been an incredible Zoo team effort and always amazing to collaborate with  @dott1718.

We will try to tell the story of progress a bit, without only explaining the end solution as we believe that the whole process of a competition is important to understand how decisions have been made.

### Early efforts
We started the competition by fitting 3-head models for R,C,V and quite quickly got reasonable results landing us somewhere in range of top 40-50 on LB. As always, we tried to understand how test data can be different, and why there is a gap to local CV. It became quite quickly clear to us that it is mostly due to unseen graphemes in test. We tried to assess how such a gap can exist, and checked how R, C and V scored individually on LB (you can figure this out quite easily by predicting all 0s for the rest, which is baseline sub). We saw that most drop comes from R and C, which is impacted by the huge effect of rare case misclassifications. So we tried something funny, which is adding additional C=3 and C=6 predictions based on the next highest indices. Just adding 310 C=3 and 310 C=6 (which reflects one extra grapheme for each) boosted us 30-40 points on LB. So we figured that this will be key in the end, and we need to generate a solution that generalizes well to unseen ones. We did also not want to rely on hardcoding too much, so tried to find a better way. These early subs with no blending at all, but this simple hardcoding, would btw still be rank 10-20 now.

### Grapheme models and fitting
Seeing reported CV scores on forums, which were all quite higher than ours, made us think that top teams are doing something different than us, which apparently was different targets. So our, first trick was to switch from predicting R,C,V to predicting individual graphemes. This also had benefits to us for not needing to care about proper loss weights of the 3 heads, different learning weights for the heads, how to deal with soft labels properly, and we could focus fully on NN fitting and augmentations. After we switched, mixing augmentations shined immediately - we replaced cutout with cutmix and then with fmix, which made it to the final models we had. Fmix (https://arxiv.org/abs/2002.12047) worked clearly better for us than cutmix, and also the resulting images looked way more natural to us due to the way the cut areas are picked. This is an example of a mixed image:



In the end, we mixed either 2 images or 3 images with 50% probability and used beta=4 to mostly do equal mixing. 

Having this grapheme model, the tricky part was to apply the model to test. At first, we tried applying the model as it is, so predicting only the known graphemes by just converting the predicted grapheme to its R,C,V components. LB score was better than for the model predicting R,C,V separately, which tells us there are not that many new unseen graphemes in public LB. To make the model work better on unseen graphemes, we applied the second trick - a post-processing routine. Surprisingly, it also improved the metric on the known graphemes as well.

The model outputs 1295 probabilities of each distinct train grapheme. For each component, e.g. C, we calculate the scores of each C=0,..,6 by averaging probabilities of the graphemes having this component. So, for C=3 it is an average of only 4 probabilities as there are only 4 graphemes with consonant diacritic of 3 in train data. For C=0 it is an average of hundreds of probabilities as it is the most common C value. The post-processing ends with picking C value with the highest “score”. The logic behind it was to treat each C value equally regardless of its frequency similar to the target metric, not to limit the model to the set of train graphemes, and pick more likely component values for the graphemes with non-confident predictions. This routine immediately gave us another 50 extra points on LB.

### Improving unseen graphemes
The third trick was about improving the predictions of unseen graphemes without hurting predictions of known graphemes too much. As most of you probably know and as elaborated earlier, the gap between cross-validation and public LB was coming from the drop of accuracy in the C component, caused mainly by C=3 and to lesser extent by C=6. It is easy to explain, as these are the rarest classes in train, and most probably public LB has at least one new grapheme with C=3 and C=6. The second largest drop was coming from the R component, while V recall was quite close between cross-validation and public LB.

The main problem with both 3-head models and specifically grapheme models is that they overfit to the seen graphemes as they are heavily memorizing them. So to close the gap a bit, we needed models which predict R and C better on unseen graphemes, which led us to fitting individual models for these 2 components. Specifically for C it was very hard to generalize and what helped a bit was to randomly add generated graphemes based on the code we found in this kernel (kudos to the author!) https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn. But, as we learnt after the end of the competition, there was way more room to improve the models this way than we actually did.

### Blending 
Trick number four was about how to blend the models together. To test the blending approaches, we recreated the hold-out sample by removing a few graphemes completely from the training part, also a couple of graphemes with C=3 and 6. We ended up doing different blending per component. But first, the models we had by the end of the competition are:
- 3 grapheme models, fitted on the whole train. The modes use adam and sgd scheduling, fitted for 80 - 130 epochs and all use fmix mixing 2-3 images.
- 1 model with 3-heads for all components. Cutout instead of fmix, sgd and 40 epochs, similar to all the following models
- 1 model for R
- 2 models for C

For R the blend is: post-processed average of grapheme models + 0.4 * average of 3-heads model and R model

For V the blend is: post-processed average of grapheme models + 0.2 * 3-heads model scaled to have equal means

For C the blend is: post-processed average of grapheme models + 15 * average of C models * C class weights

C class weights were introduced to fix the imbalanced frequencies of C, especially C=3 and 6. They were set to inverse frequencies of the class in train and normalized.

6 out of 7 models are SE Resnext50 and one is SE Resnext101. Image size was either original or 224x224. Adam scheduling with decay worked well, but SGD with scaled down every X epochs was even better. Fmix was done on the entire sample, meaning that for each image there were 2 or 3 (random with prob=0.5) random images picked from the whole training sample and mixed. 

Happy to try to answer any questions!
