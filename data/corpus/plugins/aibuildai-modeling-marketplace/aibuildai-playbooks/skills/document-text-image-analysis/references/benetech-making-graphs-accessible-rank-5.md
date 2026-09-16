# 5th place solution

Competition: benetech-making-graphs-accessible
Rank: #5
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418477

Many thanks to Kaggle and Benetech for this interesting competition where many different approaches are possible and ideas are endless. Given the strong distribution shift in the private test set, we are quite happy with our finish.

**Summary**

Our solution consists of 3 main components:
-	Synthetic data generation.
-	3-stage training of matcha models.
-	Separate process to handle scatter plot.

**1. Synthetic data generation**

Early on we realized this competition is unique in the sense that we are not limited to the available dataset. If the model struggles with certain plot patterns, we can generate a large number of plots with these patterns and as the model (matcha) is very strong, it will learn.

We started by building upon this amazing repo: https://github.com/rakutentech/chart-synthesizer.  We used both competition’s extracted data and ICDAR data as validation and try to emulate as many patterns from them as possible. For example:
-	Different fonts, tick orientations, tick styles, background colors, grid styles etc.
-	Add error bars to bar and line plots.
-	Generate histogram plots as a separate chart type.
-	Difficult line plot patterns, such as when the line starts very close to a tick mark but doesn’t touch it, the model without additional training data will very likely include this tick value in the prediction. 
-	Add blur and noise effects to reproduce the look of extracted plots.

Some examples of our generated data:



With each round of training, we would analyze the validation set, locate the patterns that give low score, and add these patterns to our chart generation code. We made the rise from 0.74 to 0.82 public LB simply by repeating this process multiple times.

With our generation code we can create as many bar, line and scatter plots as necessary for each training stage. For dot plot we sampled 10k images from the great dataset provided by @brendanartley 

**2. Three-stage training of matcha models**

Thanks to the kind sharing by @nbroad , we started the competition training `matcha-base` models and quickly reached a good score.

The training is done in 3 stages:
-	Train from `matcha-base` checkpoint as a chart classifier. The model is trained for 5 epochs, using little synthetic data and oversampling extracted data.
-	Use the weight from stage 1, continue training for 10 epochs to extract data series from all chart types. In this step we add a lot of our own synthetic data (~150k images) to combine with the original dataset.
-	Use the weight from stage 2, finetune separate models for 5 epochs with each chart type group:
    + vertical bar/dot (add 50k synthetic vertical bar plots, 5k histograms, 5k dot plots)
    + horizontal bar (add 50k synthetic plots)
    + line (add 200k synthetic plots)
    + scatter (add 30k synthetic plots)

At inference time, the chart classifier is run first and then each group of charts will be handled by its dedicated model.

At the beginning our process had only step 2 and 3, and we used model from step 2 as classifier. But then we observed that adding a lot of synthetic data decreases the performance of classifying task. As a result, training for the classifier was separated.

It is also possible to perform classifying task with a simple CNN, however from our early experiments we saw that using matcha gave a little better performance, so we sticked with its usage.

**Some training details:**
-	Learning rate 3e-5 for 1st and 2nd stages, 2e-5 for 3rd stage, using cosine schedule with warmup.
-	Adafactor optimizer.
-	Max patches 2048, max length 512.
-	Freeze first 4 layers of encoder.
-	Prediction string: `<chart_type><start>x1|y1;…;xn|yn<end>`
-	Histogram is treated as a separated chart type in training and mapped back to vertical bar at inference.
-	Dynamic rounding based on the range of data series. 

**3. Scatter plots processing**

While the mentioned above method helped us reach good results for bar, dot and line charts, we found scatter plots much harder for matcha to handle. As a result, we developed a separate scheme only for scatter:

-	Plot area and tick label detection: use outputs from [CACHED](https://github.com/pengyu965/ChartDete) .
-      Textline Rotation : use MobilenetV2 backbone to train angle classification task.
-	Textline OCR: use [vietocr](https://github.com/pbcquoc/vietocr) ’s seq2seq model to train textline reading task. We modified the encoder to use ResNeXt50 with dilated convolution.
-      Scatter marker detection: use Mask-RCNN with [CoaT](https://github.com/mlpc-ucsd/CoaT) backbone for this task. We trained the model with detectron2. 
-      Final values of each marker are deduced from marker position, tick label position and tick values.

This process helped us score 0.09 public and 0.28 private for scatter and is the key for us to survive the private test set.

Thank you for reading and please let us know if you have any questions.

Edit: Inference notebook for our solution now available: https://www.kaggle.com/code/qdv206/benetech-5th-place-inference
