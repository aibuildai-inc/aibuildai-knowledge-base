# 5th place solution

Competition: bengaliai-cv19
Rank: #5
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136129

Thanks at Bengali.ai and kaggle for hosting this interesting competition as it was more to it than simple ensembling of computer vision models. Thanks to the authors and contributors of pytorch, pytorch-lightning, apex and pytorchcv for making my life easier.

**Acknowledgements &amp; notes on GM**

I am humbled to finally become competition grandmaster. I learned a tremendous amount of tricks in the recent two years and it would not have been able if it wasn’t for all this generous sharing of top solutions and extremely talented teammates I was lucky to have along the way. 


### Short Summary

My solution is a simple bag (different seeds) of the same model seresnext50 with custom head. Major boost in LB score came from redesign of consonant diacritic target.  Minor LB improvement from postprocessing tweaks such as thresholding and finding closest train examples. 


### Preprocessing

I did not resize the image, the only thing I did was normalize each image by its mean and std, since I experienced a good regularization from that in previous competitions.

### Architecture &amp; Training

**Backbone**
I used a plain seresnext50 but adjusted the very first layer to replace resizing the image and account for single channel. I did that by changing input units and reducing stride from (2,2) to (1,2).

```
from pytorchcv.model_provider import get_model as ptcv_get_model

backbone = ptcv_get_model('seresnext50_32x4d', pretrained=True)
backbone = backbone.features
backbone.init_block.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 2))
```

**neck**

same as in maciejsypetkowski solution https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543 


**heads**

- 3 heads, each for consonant, vowel and root and 
- auxiliary head for grapheme with arccos loss

So in total my used architecture looks like this:




**Training**

- train on all data
- 4 cycle cosine annealing with augmentation increase
  - cycle 1 &amp; 2 cutmix
  - cycle 3 &amp; 4 cutmix, cutout, scale, translate, rotate
- Adam 

**Losses**

- root loss: CrossEntropy
- consonant loss: Multi label Binary Crossentropy
- vowel loss : CrossEntropy
- grapheme loss: ArcCos + CrossEntropy

- total loss = root loss + consonant loss + vowel loss + grapheme loss


### The Magic (jôfôla) 
Now to the key aspect of my solution. I realised all models perform poorly on consonant diacritic 3 (I abbreviate with cons3 in the following) and especially confuse it with cons4. And with poorly I mean game changing poorly: 89% recall compared to 99% for the other consonant classes. At first I just fought the symptoms (instead of the root cause) as several other teams did, e.g. by transferring low confident predicts from high frequent class cons4 to low frequent class cons3 and hence leveraging the definition of macro recall metric and that gave a good LB boost (0.9900 -&gt; 0.9915). That might have been enough for gold but I wanted to improve my chances. So I wanted to understand why the model did so poor on this one and read through discussions, wikipedia and other domain knowledge related things. Luckily, I found 2 things:
- Wikipedia: As the last member of a conjunct, য jô appears as a wavy vertical line (called যফলা jôfôla) to the right of the previous member: ক্য "kyô" খ্য "khyô" গ্য "gyô" ঘ্য "ghyô" etc. In some fonts, certain conjuncts with যফলা jôfôla appear using special fused forms: দ্য "dyô" ন্য "nyô" শ্য "shyô" ষ্য "ṣyô" স্য "syô" হ্য "hyô".
 
- The consonant diacritics are itself composed from lower level components
  - 0: []
  - 1:  ['ঁ']
  - 2: ['র', '্']
  - 3: ['র', '্', 'য']
  - 4: ['্', 'য']
  - 5:['্', 'র']
  - 6: ['্', 'র', '্', 'য']
 
So I realized that not only 3,4 and 6 have a jô component which might have the special jôfôla form but also that the special forms are in train for **only** cons4 but **not** for cons3, and thats why the models are so bad on cons3. My solution was the following. I recoded the cons classes into a multilabel classification problem using the following function:

```
cons_components = ['ঁ', 'য', 'র','্']

def is_sub(sub, lst):
   ln = len(sub)
   return any(lst[i: i + ln] == sub for i in range(len(lst) - ln + 1))

def label2label_v2(label):
   res = np.zeros((6,),dtype=int)
   components = cons2components[label]
   for i,item in enumerate(cons_components):
       if item in components:
           res[i] = 1
   if is_sub(['্', 'য'], components):
       res[4] = 1
   if is_sub(['্', 'র'], components):
       res[5] = 1
   return res

```

That function re-codes the 7 consonant class labels into multilabel 6 dimensional vectors

```
0 -&gt; [0, 0, 0, 0, 0, 0]
1 -&gt; [1, 0, 0, 0, 0, 0]
2 -&gt; [0, 0, 1, 1, 0, 0]
3 -&gt; [0, 1, 1, 1, 1, 0]
4 -&gt; [0, 1, 0, 1, 1, 0]
5 -&gt; [0, 0, 1, 1, 0, 1]
6 -&gt; [0, 1, 1, 1, 1, 1]
```

That schema not only enabled to learn special forms of cons3 from cons4 but also enables to tune via thresholding (i.e. when to set a continuous prediction to 0 or 1) how the model separates the 6 classes. E.g. by setting a very low threshold on the 2nd logit more predictions would be set to 1 and hence more predictions would move from cons4 to cons3 as these two classes only differ in logit2. Purely changing the target for consonant diacritic in this way improved Public LB from 0.990 to 0.993 !


### Postprocessing

- I adjusted threshold for binarizing consonant diacritic. Best was a threshold of 0.1 for all logits classes. I tried not dare to temper which each logit threshold individually as I was afraid of overfitting. Result on Public LB: 9930 -&gt; 9937 
- I used cosine similarity to find closest train sample for root and vowel when top1 probability &lt; 90% and use its label. bestfitting did the same in the 1st place solution of Human Protein competition) as can be found under https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109 Result on Public LB: 9937 -&gt; 9942 


### Choosing subs
I tempered around with binarization and metric learning thresholds and purely checked LB hence I was aware of a high risk to potentially overfit to LB. So I chose my best sub and a failsafe sub that I felt was the least tempered with but still doing ok. (Both were gold on private at the end)

Cheers.
