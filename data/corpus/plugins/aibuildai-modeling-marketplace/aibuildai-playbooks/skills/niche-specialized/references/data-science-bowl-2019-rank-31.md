# Single Model - 1.5 Transformers - 31st place

Competition: data-science-bowl-2019
Rank: #31
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127544

This was a very challenging and fun competition. I want to thank the sponsor, Kaggle, and of course all the amazing competitors!

I also want to give a shout out to the Kaggle/Google engineers.  I've been incredibly impressed with the website/kernels.  I am currently a full stack engineer at Amazon so I know how hard it is to pull that off, so thanks for making such an amazing product!

## Data Preparation
I did some minor edits to event codes.  I broke up 4020, 4010, 4025 into whether correct was true or false, so I ended up with 40201, 40200, 41001, 41000, 40251, 40250.

I transformed the data into "histories".  Basically a history is all of the data leading up to the target assessment.  I then processed these histories into a large numpy array.

Due to some histories being enormous due to shared devices I decided to take the last X game sessions per target assessment and also the last Y events per game session.  This made sense to me as the recent data should be more important and it is an easy way to deal with shared devices.

I found that x = 80 and y = 100 gave the best results, so I ended up with a sparse np array:
(# histories, 80, 100, # features)

I added "blank" categories to the title, event_code, and accuracy group embeddings.  This informed the model that these did not exist.  (History was shorter then 100 events or shorter than 80 game sessions.)  I tried masking the input to the transformer but it killed the performance and the score decreased.

## Features
You can see the features below in the model diagram.  The Assess Target Title and Assess Target Time are fed into every event.  I did try inputting these once at the end of the model but the performance got slightly worse.  I also tried inserting game session features into the game session embedding but none of the features I tried helped.

Near the end of the competition I played around with adding in the OOF models' predictions and the models' prediction groups.  This seemed to help a lot on Local CV but not as much on Test.  I think perhaps I was doing something wrong with how I was then creating these values for the Test Assessments.

## Model
My original idea was to use a double transformer network.  One transformer for the events of a game session and then use those outputs to have one transformer take in each game session embedding.  This did work but I discovered it was better (and much faster) with the first transformer being a "Zero Head", which means I just removed the attention part and left in the shared FC layers:

 &gt;events2 = self.linear2(self.dropout1(F.relu(self.linear1(events))))
 &gt;events = events + self.dropout2(events2)
 &gt;events = self.norm2(events)

One key idea came from NLP where they tie the word embeddings in the input and in the output to improve generalization.  I did this with the accuracy group and saw a nice bump in QWK.

.png?generation=1579876333416350&amp;alt=media)

## Noisy Labels
This was one of the more interesting ideas that I tried.  If you consider the fact that 3-5 year olds are incredibly noisy in general you can view the labels as being fairly noisy.  I read a bunch of papers on dealing with noisy labels and they all basically dealt with the model (or a second model) learning the noise.  I decided to save the OOF predictions and then blend these with the actual targets when training new models hoping the single model would be able to learn about some of the noise patterns.  Turned out this was very difficult to tune correctly as it seem to leak into my 5 fold CV and also made the train loss hard to reason about.  Basically it made it very easy to overfit to the train data.

I'm also not sure if the way I did the OOF predictions was the best.  I would save it after every 5 fold run and then would just average the predictions from all of the past.  I think now that this may have increased the confirmation bias and I would have better off with just getting some initial predictions from models that were not blended with new targets and sticking with those.

In the end it did boost my private test score from .550 to .554.  My best blend was 85 epochs starting from all actual targets to 50/50 at epoch 50 and then increasing back in the actual targets.  The blend was linear changing by 1% every epoch so old/new: 1/0 -&gt; .5/.5 -&gt; .85/.15

Here was the main paper that I got this idea from.  It's on pseudo labeling but I think it applies just as well to dealing with noisy labels.  I had mixup augmentation on my todo list as I think it would have greatly enhanced the Noisy label technique, but I never got to try it.

&gt; Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning
&gt; Eric Arazo, Diego Ortego, Paul Albert, Noel E. O’Connor &amp; Kevin McGuinness
&gt; https://arxiv.org/pdf/1908.02983.pdf

## Code
I'm working on cleaning up the code and hope to make it public soon.  Thanks for reading!
