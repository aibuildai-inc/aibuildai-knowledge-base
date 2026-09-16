# 13th place brief writeup (public 107th)

Competition: bengaliai-cv19
Rank: #13
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136116

updated 2020/03/19

----

Thanks to the hosts and helpful discussions.
Especially, I really appreciated @hengck23's kind sharing.

I am so lucky to get gold.
I expected random-split public/private and did nothing special for unseen grapheme.

It seems postprocessing for macro recall hack is really important.


## Model
- modified SE-ResNeXt50. 2-stride conv is replaced by 1-stride-conv+maxblur-pool
   basically following @hengck23 's [advice](https://www.kaggle.com/c/bengaliai-cv19/discussion/123757#757358)
- phase1: trained with stem-conv(stride=2), phase2: replace stem-conv with 3 convs(stride=1)
- Adam + ReduceLROnPlateau, about 200 epochs
- 5 fold ensemble
## Loss
- CrossEntropy for root, vowel, consonant
  - 2 out of 5 folds are finetuned using ohem loss
- Multilabel BinaryCrossEntropy to classify decomposed grapheme exists or not
  - graphme (or root or vowel or consotant) can be decomposed as shown in the figure. 61 possible decomposed parts exist (except '0' part). below is my exact code to make multihot label.

```
# load data
train = pd.read_csv(C.datadir/'train.csv')
train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values.astype(np.int64)

# components label
parts_pre = pd.read_csv(C.datadir/'class_map.csv').component
parts = np.sort(np.unique(np.concatenate([list(e) for e in parts_pre])))
parts = parts[parts != '0']  # 0 has no meanning
print("parts:", parts)  # 61 parts shown

train_labels_comp = []
for grapheme in train['grapheme'].values:
    train_labels_comp.append([part in list(grapheme) for part in parts])
train_labels_comp = np.array(train_labels_comp).astype(np.int64)
print("train_labels_comp.shape", train_labels_comp.shape)
if True: # debug
    print("train_labels_comp", train_labels_comp[0].tolist())
    print("train_labels_comp", train_labels_comp[1].tolist())
    print("train_labels_comp", train_labels_comp[2].tolist())
```

## Preprocess and Augmentation
100% same as @hengck23 's [advice](https://www.kaggle.com/c/bengaliai-cv19/discussion/132898#763633)
Preprocess :  just resizing with cv2.INTER_AREA
Augmentation : OneOf(basic augmentations) + DropBlock
## Postprocess
- prediction for decomposed grapheme is not used
- **maximize expected recall by
  argmax( softmax(logits) / np.power(class_count, 1) )**. note that final sub use factor of 1.15 for grapheme root(no difference for private lb).
  - there may be better threshold optimization since confusion matrix still looks asymmetry after above optimization. But I gave up further improvement due to low sample size.
  - below is plot of "macro recall scores V.S. factor n" for my model (which is not final sub). You can confirm that the peak is around factor=1. You can also find that the improvement is not huge unlike private LB.


-----
**What I learned during competition ( note to self )**
*how to use pytorch lightning / how to freeze specified layer / how to ensemble / concept of snapshot ensemble / ohem loss, class-balanced loss / hengck23's model surgery method / maxblur-pool and its implementation / keep high resolution may be important for low resolution image / interpolation method for resizing is sometimes important / how to implement some augmentation methods (mixup, cutmix, gridmask) / public &amp; private is not always split randomly / vast.ai is low-priced compared to GCP*
