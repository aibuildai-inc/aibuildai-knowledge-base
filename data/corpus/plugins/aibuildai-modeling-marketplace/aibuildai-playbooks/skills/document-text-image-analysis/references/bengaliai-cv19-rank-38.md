# Solution 38, some lessons learned

Competition: bengaliai-cv19
Rank: #38
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136075

First of all, let me thank my team mate Marios (Kazanova). Without him I'm not sure I would have a medal at all here.  Second, I want to thank all those who shared so much.  This was a goldmine for people with little experience in computer vision like me.  I wish I had paid more attention on some of it, for instance discussion about unseen graphemes in test, or predicting graphemes rather than components as a secondary target.  Last, but not least, many thanks to Bangali AI sponsor and Kaggle for providing a challenging image classification problems with rather small size dataset.  Hardware (access to a number of recent GPU) wasn't the main deciding factor here.  

I am a bit disappointed to have dropped out of gold after being up to top 7th on public LB of course.  At the same time, my first sub is only from 13 days before competition end.  And I could team with Marios who had started earlier.  Still, issue with little time is that you have to be right in selecting what to try and what you decide to no try. And given training computer vision models takes time, you easily lose a day or two on ideas that don't work.  I won't enter computer visions less than 2 weeks before end again.  First lesson learned.

We didn't looked at which targets where hard to predict.  In general, I didn't look at data enough.  Second lesson.

 I mentioned some of what we didn't try unfortunately, here is what we tried, with more or less success.

Before teaming I worked on 64x64 images, and a bit on 128x128 images with seresnext50.  Small image sizes leads to high epoch throughput and allow for lots of experiments.  I didn't work on image augmentation and focused on learning pytorch, understanding losses and image preprocessing.  When I teamed, Marios had a efficientnet-b4 model with a good augmentation pipeline.  He used cutout, grid mask, mixup, albumentation, and found that cutmix wasn't effective.  He also used ohem loss, and cosine lr scheduler.  We didn't revisit this except for ohem loss after we teamed.

After teaming we explored ways to add diversity, and ways to combine as many models as possible in the scoring kernel.  In hindsight it also helped overfit a bit more to the public test data.  Our CV LB score gap was rather stable, which also explains why we did't explore more the possible public/private distribution discrepancy.  Third lesson learned, pay more attention to this.

We used an average of 14 efficientnet-b4 models predictions in the end.  Our scoring kernel could handle up to 18 models within 2 hours.  For each model we averaged weights of several checkpoints.  Models differ by:

- input data, original size, or crop resize with same image ratio
- fold or full train data
- loss, weighted loss for R, V, C, or weighted loss per class within each of R, V, C.
- ohem or unmodified cross entropy loss
- use of external data or not
- postprocessing 

We looked at postprocessing predictions the last few hours before end.  It was clearly promising, and one sub did improve a bit over our best blend.

We only looked at external data, BanglaLekha-Isolated two days before end of competition.  We predicted targets using our best blend, then took the majority class and checked the mapping  by visual inspection between the BanglaLekha paper and the class_map csv.  Our mapping was wrong in few case.  Then we assigned the fixed for R target and the 0 target for V and C.  CV and Lb scores were not really better but it added dieversity to the blend.  Here again, we had to crop resize with same image ratio.

All in all I learned a amazing lot in a short time.  This is where Kaggle is great, you can get a glimpse of state of the art practice in a very short time.  Again, this is only because of sharing from the community members.  I'm now looking forward to next computer vision competition to be honest!

I hope I haven't missed too much, especially on what Marios did before we teamed.  I'll let him expand/correct me if need be.
