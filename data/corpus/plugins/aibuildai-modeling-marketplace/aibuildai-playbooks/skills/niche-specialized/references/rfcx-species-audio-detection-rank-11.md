# 11th place, The 0.931 Magic Explained: Image Classification

Competition: rfcx-species-audio-detection
Rank: #11
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220304

First of all, thanks to the host, for providing yet another very interesting audio challenge.  The fact that it was only partly labelled was a significant difference with previous bird song competition.

Second, congrats to all those who managed to pass the 0.95 bar on public LB.  I couldn't, and, as I write this before competition deadline, I don't know why.  I tried a lot of things, and it looks like my approach has some fundamental limit.

Yet it was good enough to produce a first submission at 0.931, placing directly at 3rd spot while the competition had started two months earlier.

This looked great to me.  In hindsight, if my first sub had been weaker, then I would not have stick to its model and would have explored other models probably, like SED or Transformers.

Anyway, no need to complain, I learned a lot of stuff along the way, like how to efficiently implement teacher student training of all sorts.  I hope this knowledge will be useful in the future.

Back to the topic, my approach was extremely simple:  each row of train data, TP or FP, gives us a label for a crop in the (log mel) spectrogram of the corresponding recording.  If time is x axis and frequency the y axis, as is generally the case, then t_min, t_max gives bounds on x axis, and f_min, f_max gives bounds on the y axis.

We then have a 26 multi label classification problem (24 species but two species have 2 song types.  I treated each species/song type as a different class).  This is easily handled with BCE Loss.

The only little caveat is that we are given 26 classes (species + song type) but we get only one class label, 0 or 1 per image.  We only have to mask the loss for other classes and that's it!

I didn't know it when I did it, but a similar way has been used by some of the host of the competition, in this paper (not the one shared in the forum): https://www.sciencedirect.com/science/article/abs/pii/S0003682X20304795

The other caveat is that the competition metric works with a label for every class.  Which we don't have in train data.  However the competition metric is very similar to a roc auc score per recording: when a pair of predictions is in the wrong order, i.e. a positive label has a prediction lower than another negative label prediction, then the metric is lowered.  As a proxy I decided to use roc-auc on my multi label classification problem.  Correlation with public LB is noisy, but it was good enough to let me make progress without submitting for a while.

What worked best for me was to not resize the crops.  It means my model had to learn from sometimes tiny images.  To make it work by batch I pad all images to 4 seconds on the x axis.  Crops longer than that were resized on the x axis.  Shorter ones were padded with 0. One thing that helped was to add a positional encoding on the frequency axis.  Indeed, CNNs are good at learning translation independent representations, and here we don't want the model to be frequency independent.  I simply added a linear gradient on the frequency axis to all my crops.

For the rest my model is exactly what I used and shared in the previous bird song competition: https://www.kaggle.com/c/birdsong-recognition/discussion/183219  Just using the code I shared there was almost good enough to get 0.931.  The only differences are that I add noise as in the first solution in that competition.  I also did not use a no call class here, nor secondary labels.

For prediction I predict on slidding crops of  each test recording and take the overall max prediction.  This is slow as I need to do each of the 26 classes separately.  This is also maybe where I lost against others: my model cannot learn long range temporal patterns, nor class interactions.

With the above I entered high with an Efficient B0 model, and moved to 0.945 in few submissions with Efficientnet B3 . Then I got stuck for the remainder of the competition.  

I was convinced that semi supervised learning was the key, and I implemented all sorts of methods, from Google (noisy student), Facebook, others (mean student).  They all improved weaker models but could not improve my best models.

In the last days I looked for external data with the hope that it would make a difference. Curating all this and identifying which species correspond to the species_id we have took some time and I only submitted models trained with it today.  They are in same range as previous ones unfortunately.  with a bit more time I am sure it could improve score, but I doubt it would be significant..

For matching species to species id I used my best model and predicted the external data. It would be interesting to see if I got this mapping right.  Here is what I converged to  :

0                        E. gryllus
1         Leutherodactylus brittoni
2           Leptodactylu albilabris
3                          E. coqui
4                       E. hedricki
5                 Setophaga angelae
6          Melanerpes portoricensis
7                  Coereba flaveola
8                       E. locustus
9                Margarops fuscatus
10          Loxigilla portoricensis
11                 Vireo altiloquus
12                 E. portoricensis
13                Megascops nudipes
14                     E. richmondi
15             Patagioenas squamosa
16    Eleutherodactylus antillensis
17                  Turdus plumbeus
18                      E. unicolor
19               Coccyzus vieilloti
20                  Todus mexicanus
21                    E  wightmanae
22         Nesospingus speculiferus
23          Spindalis portoricensis

The picture in the paper shared in the forum  helped to disambiguate few cases: https://reader.elsevier.com/reader/sd/pii/S1574954120300637  The paper also gives the list of species.  My final selected subs did not include models trained on external data, given they were not improving.

This concludes my experience in this competition. I am looking forward to see how so many teams passed me during the competition.  There is certainly a lot to be learned.

Edit.  I am very pleased to get a solo gold in a deep learning competition.,  This is a first for me, and it was my goal here.

Edit 2:  The models I trained last day with external data are actually better than the ones without. The best one has a private LB of 0.950 (5 folds). However, they are way better on private LB but not on public LB.  Selecting them would have been an act of faith.  And late submission show they are not good enough to change my rank.  No regrets then.

Edit 3  Using Chris Deotte post processing. my best selected sub gets 0.7390 on private LB.  It means that PP was what I missed and that my modeling approach was good enough probably.  I'll definitely look at test prediction distribution from now on!
