# 1st place solution: post-processing

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #1
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160986

Firstly, a big thanks to Kaggle for constantly delivering top-notch competitions. Competitions doesn't always end smooth (cf. Deepfakes 😄) but there is no other data science platform so rich in learning. Another thanks to Jigsaw for the interesting, multi-lingual, shakeup-free ride!  
Also congrats to the other medalists, and to my talented team-mate @leecming!

Since other competitors were asking about our post-processing technique, I dedicate this post to explain it in more detail. 

To see the post-processing in action, have a look at the related [notebook](https://www.kaggle.com/rafiko1/1st-place-jigsaw-post-processing-example)
## Post-processing: intuition

The intuition of the post-processing is as following: we consider the ***trend*** of subsequent submissions of a specific language (e.g. Russian) for each example in the test dataset. If the trend of that example is positive i.e. going up, we ***nudge*** the example further in the positive direction. And vice versa - if the trend is negative i.e. going down, we nudge the example further in the negative direction.

We measure the trend by taking the differences of all subsequent submissions for the specific language and averaging those differences. The nudge that we then give to the new submission is based on a predefined ***weight***, typically we choose a weight of 1 or 1.5. 

## Post-processing: pseudo-code
In an attempt to pseudo-code the technique, given:

```
weight = predefined weight (typically 1 or 1.5)
pred_best = current best predictions on LB
diff_avg = average of differences of consecutive subs (trend)
```

Then for each example in test of the specific language (e.g. Turkish):

```
if diff_avg &lt; 0: # negative trend
    pred_new = (1+weight*diff_avg)*pred_best  # nudge downwards
else: # positive trend
    pred_new = (1-weight*diff_avg)*pred_best + weight*diff_avg # nudge upwards
```

Note: I'm not an expert on PP, so I assume this technique can be further optimized. The boost we got from it was relatively small (albeit significant) compared to the other methods we implemented.
