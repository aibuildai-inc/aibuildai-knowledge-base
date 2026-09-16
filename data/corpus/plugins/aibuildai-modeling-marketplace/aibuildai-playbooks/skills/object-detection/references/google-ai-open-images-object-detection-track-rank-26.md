# 26th place solution [0.38 private LB]

Competition: google-ai-open-images-object-detection-track
Rank: #26
Source: https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64901

# Software
* Ubuntu 16.04
* tensorflow
* yolo3: https://github.com/qqwweee/keras-yolo3
* faster rcnn: object detection in official models of tensorflow models

# Main approach
* data balancing
* tricks of learning rate
* study the rule of scoring

# Code
* https://github.com/rabienrose/GoogleAIOpenImg2018
* Include the necessary files for my project from tensorflow models and keras-yolo3

# Experience
- One of my friends suddenly asked me to join the challenge one day. I have never involved in any this kind of competition and I have not done any object detection projects before. I only did some classification projects before.Just reading the rule for the competition make me uneasy. But, I like the feeling of pushing myself to the limit, so I started to download the tensorflow models for github.
- I first tried the fast-rcnn model, before it was said the most powerful model current. But it was too slow to train, about 600ms per image with my 1080ti. I only trained with about 20,000 images for couple of epoch. The first submission is very bad, only something like 0.00002. The main result not because the training is bad but because some bugs: I mistake the order of xmin, ymin and so on. After fixed this, I got score like 0.02.
- The speed of training is a problem, with that speed, I have no time to adjust the model and study the result. So, I decided to move to yolo3. And I also do some data balancing: get rid of images that only contain the very common objects like person, face. Finally, I chose 200,000 images out of 1700,000 train images.
- It was about 4 hours to train one epoch, I increase the learning rate a little bit if I find the loss enters the platform status. Otherwise I just linearly decrease the learning rate as epoch increase.

# Other configs
* Confidence threshold: 0.01
* Expand the classes based on the hierarchy after prediction, not before training
* No ensemble (I really did not have time to train multiple models)
