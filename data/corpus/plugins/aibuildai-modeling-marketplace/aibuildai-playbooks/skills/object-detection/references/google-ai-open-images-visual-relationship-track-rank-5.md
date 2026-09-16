# 5th place solution overview

Competition: google-ai-open-images-visual-relationship-track
Rank: #5
Source: https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64826

[tldr; version (Twitter thread)][1]

I trained two models - one for the 'is' relationship and one for all the others. For the 'is' relationship I used a sigmoid output layer with 5 units to predict whether the item belongs to one or more of the 5 'is' categories.

For all the other relationships I presented the model with a of items and asked it to predict whether they are related or not (binary classification).

I fed 128x128 crops to the models. For the 'is' relationship I blew up the detections to occupy the entire image. For all the remaining relationships I took crops without distorting the aspect ratio. Further to that, I masked out the area outside of detections.

I sent the crops through a CNN, inception v4, and I concatenated the outputs with information I derived from the coordinates, feeding the coordinates as well. I calculated intersection over union between detections, relative distance and relative size. Further to that, I also used embeddings for the relationship and both of the labels. For the 'is' model I don't include some of the information that does not make sense in this context (IOU, relationship embedding, etc).

I used the [fastai library][2] which is built on top of pytorch. This gave me a lot of flexibility. I developed and ran the solution in Jupyter notebooks. Despite not being able to give the competition as much time as I would like, this combination of tools allowed me to test quite a few hypotheses. For instance, turns out there is a lot of information contained in the coords, it seems important how you pose the question to be answered (though more investigation into this could be beneficial), etc.

Once I have the predictions I scale them by relationship frequency in the train set. I also scale the probabilities by the initial detection scores coming from yolov3. I provide an overview of how I trained the model I use for detection [here][3].

For the vrd track, I did all the work on a machine with a Ryzen 5 CPU, 16 GB of RAM and a single 1080TI. I obtained and resized the crops before the training and stored them in a mongo database.


  [1]: https://twitter.com/radekosmulski/status/1035496977847517184
  [2]: https://github.com/fastai/fastai
  [3]: https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64734#379875
