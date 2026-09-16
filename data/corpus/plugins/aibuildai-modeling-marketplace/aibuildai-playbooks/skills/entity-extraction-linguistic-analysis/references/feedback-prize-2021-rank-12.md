# 12th place solution - Stretching short 'predictionstring's

Competition: feedback-prize-2021
Rank: #12
Source: https://www.kaggle.com/c/feedback-prize-2021/discussion/313833

First of all, I would like to thank Kaggle and the organizers for hosting this competition, and everyone who shared useful methods and information.

I've refered to a lot of shared ideas in this competition.

## Solution

As the published method, I've handled  the task as NER.
My solution consists of below two parts.

1. Weighted ensemble of 5 models
2. Several postprocess (including stretching short 'predictionstring's)  

### 1. Weighted ensemble of 5 models
My models are based on @Abhishek's code.

I took a weighted average of below 5 models.
Each model is trained on 5 folds and chosen some folds in submission.
|model|weight|the amount of used folds|
| :---: | :---: | :---: |
|  longformer-large |  0.175  |  3  |
|  funnel-large |  0.175  |  3  |
|  deberta-large(seed:77)  |  0.15  | 3 |
|  deberta-large(seed:456) |  0.15  | 2 |
|  deberta-xlarge  |  0.35  |  5  |

### 2. Several postprocesses
Mainly, I used below 3 postprocesses.
① <u>Stretching short 'predictionstring's（public+0.007/private+0.005)</u>
I stretched short 'predictionstring's slightly.
 The thresholds are as follows:

|discourse_type|threshold1<br>(length/degree)|threshold2<br>(length/degree)|threshold3<br>(length/degree)|threshold4<br>(length/degree)|
|:----:|:----:|:----:|:----:|:----:|
| Lead | 7~13 / +6  |  14~19 / +12 | 20~30 / +14| -  |
| Position | 5~15 / +3  |  16~20 / +2 | -| -  |
|  Claim | 1~5 / +1  |  6~10 / +2 | 11~20 / +4  |  - |
|  Counterclaim | 5~24 / +4  |  25~37 / +5 | - | -  |
|  Rebuttal | 2~4 / +1  |  5~13 / +5 | 14~21 / +7  | 22~27 / +8 |
|  Evidence | 17~20 / +11  |  21~23 / +14 | 24~29 / +17  | 30~36 / +20 |
|  Concluding Statement | -  |  - | -  | - |

Basically, I increased the length to be stretched in proportion to the length of the 'predictionstring'.
This process increased TP while decreasing FP, finally improving the public/private by 0.007/0.005.

② <u>Improvement of [link_evedence](tensorflow-longformer-ner-postprocessing/notebook)（public+0.003/private+0.003)</u>
The points are following.

・Modifying 'jn'
'link_evidence' used '-1' to distinguish between each 'predictionstring', 
so the result of 'link_evidence' contained '-1' while ground truth don't include '-1'.

Therefore I modified 'jn' which is used to concatenate several 'predictionstring's in 'link_evidence' so that the result of 'link_evidence' don't contain '-1'.
```
#before
def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

#after
def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end] if x !=-1])
```

・Applying it on other discourse types
I applied this method on other discourse types excluding 'Claim'.

・Optimizing thresholds
＝＝＝＝
The improvement increased public/private by +0.003 /+0.003.(excluding the effect of the default 'link_evidence')

③<u>Removing a capital letter on the end of the 'predictionstring'（The effect is small)</u>

Looking at the predictions, 
I found that the end of prediction is sometimes a capital letter(e.g. 'This'). 
So, if the capital letter in the end of prediction is not a proper noun, I removed it.

## What didn't worked well for me
・multi-task training using 15 topics
・concatenating last N layers of output of BERT
・several losses(Focal Loss, Dice Loss)
・large batch size

## Resourse

Colab pro+ （Mainly, V100)
GCE preemptive instance(A100)

## Inference code
The original code is [here](https://www.kaggle.com/kurokurob/12th-place-solution-original).
After the end of competition, I found some errors in the original code, so I fixed them.
The modified code is [here](https://www.kaggle.com/code/kurokurob/12th-place-solution-error-fix).
<br>
Thanks for reading!
