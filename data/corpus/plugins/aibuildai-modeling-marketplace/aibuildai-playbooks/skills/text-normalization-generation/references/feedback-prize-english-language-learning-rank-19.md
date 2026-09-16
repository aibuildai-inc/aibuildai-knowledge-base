# 22nd Solution Summary

Competition: feedback-prize-english-language-learning
Rank: #19
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369533

I was expecting a large shakeup but I did not expect to jump that way!

Here is my part of the solution. I teamed late with Dieter with the hope that adding diversity would help. I also knew he would not be overfitting to public LB which made teaming risk free. Overfitting is always the danger when we have small data like this. The final shakeup proves it was a real danger unfortunately.

It turns out that Dieter model are quite diverse from mine, mostly because he used SVR on top of embeddings as part of his solution while I didn't.

I did not have much time for this comp given I was busy on other competitions outside Kaggle (MICCAI then NEURIPS). Therefore I went for well known techniques. My solution is a mostly pseudo labeling + knowledge distillation. It was developed rather quickly, with a first 2weeks phase 2 months ago where I ensembled models based on public LB score (don't do that!), then last week of comp where I used CV to guide my work. CV based ensembling only beat ad hoc ensembling last day of competition for us! It made sub selection choice easy: our best CV was best public LB and turned out to be best private LB too.

Here is what I did on top of a standard baseline. 

- I used variants of deberta model only.
- In order to fix the label distribution imbalance between fb1 data and fb3 data I decided to round predictions on fb1 of a reasonable model around the fb3 label values. Then I weighted fb1 samples using the frequency ratio of their rounded predictions. With sample weighting pseudo labeling proved to improve CV score.
- I used pseudo labels ( prediction on fb1 data) for knowledge distillation. I ran two rounds of knowledge distillation. At each round I use previous models (round 0 is models trained on fb3 data only) to predict on fb1 data and add this to fb3 data with its original labels. Keeping fb3 labels helps avoid confirmation bias.
- models for first and second round of knowledge distillation have deberta architecture + a simple dense regression head
- for first level models (trained on fb3 data only) I also used some models with ordinal regression (see my petfinder writeup for a definition)
- Final ensemble is a blend of 4 models on my side plus one model from Dieter. I used Huber Regressor for an unknown reason to find the weights. I usually use Ridge Regression for that, not sure what I thought. I probably didn't sleep enough last night of the competition, LOL.

That is all I could do in the time I got. Dieter spent even less time than me on this competition. We are happy with surprisingly good end result given we were 700 or so on public LB. Teaming was good too as I would be 36th alone.
