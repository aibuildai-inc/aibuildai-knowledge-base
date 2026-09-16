# brief summary of 2nd place

Competition: open-images-2019-visual-relationship
Rank: #2
Source: https://www.kaggle.com/c/open-images-2019-visual-relationship/discussion/111361

Congrats to all the winners, and thanks to competition organizers for this interesting competition again.

I used almost same architecture as [last year](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64651). 
So I'll summarize only difference.


# Model 1: object detection
I made cascade-rcnn model using [mmdetection](https://github.com/open-mmlab/mmdetection).

mAP for 57 classes improved a lot, more than 0.1 compared to last year yolo model.




# Model 2: visual relationship

## 2-1: relation 'is'
I made 3 models for this part, and then I made ensemble of them.


### 2-1-1: relation 'is' (2 stage model)
This is the model I used for relation 'is' last year.


### 2-1-2: relation 'is' (1 stage model)
I made cascade-rcnn model which detect 42 'is-relation' classes.
This model is almost same as ['toshif' explained last year](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64642).

### 2-1-3: relation 'is' (1 stage model with material head)
I added 'material' detection head to cascade-rcnn.
This model predict Bounding Box and class and material at the same time.



Results:

|model  |public  |private  |
|---|---|---|
|2-1-1  |0.07523  |0.07264  |
|2-1-2  |0.08332  |0.08075  |
|2-1-3 |0.08191  |0.07948  |
|ensemble |0.08514  |0.08232  |

I expected 2-1-3 to have better score...



## 2-2: Triplet Relationships
Base model is almost same as I shared [here](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64651#380288) for this part.


I made expert models which only in charge of small sample class, and made ensemble of them with weighted average of their probability.

This is the result AP for validation data:

|class |grand truth BB |predicted BB |
|---|---|---|
|at |93% |31% |
|on |92% |32% |
|holds |89% |54% |
|plays |94% |58% |
|interacts with |82% |45% |
|inside of |72% |37% |
|wears |94% |55% |
|hits |55% |57% |
|under |50% |20% |
|mAP without hits/under |88% |45% |
|mAP |80% |43% |

For grand truth BB pairs, this relationships prediction model has very high accuracy.
mAP without hits/under which have very small samples is 88%!

# Model 3: Final Score Prediction
I did not used Light GBM for this part.
I just used simple formula.

`Final Score = Object1Score x Object2Score x RelationsipScore`



This year, my LB score improved to 0.38818 from last year score 0.23709.
Most of this improvement comes from object detection improvement.

It seems that good object detection is the most important part of this competition.





BTW, I became GM as of this competition. 
I'd like to thank to my previous team mate. 
I learned lots of things from them and I could not be GM without them.
Thank you, Carl, Little Boat, KazAnova, Ahmet, Kohei-san, Akiyama-san, owruby!
