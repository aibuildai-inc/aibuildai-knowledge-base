# 4th place (brief) tips

Competition: talkingdata-adtracking-fraud-detection
Rank: #4
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56243

I would like to start via praising my teamates for an excellent effort and congratulate the winners for  an intense last-days' race. 

Also thank you to all the people who shared code and ideas - they made it a great competition (with the exception of the latest high-scoring kernels) . Special thanks to [anttip][1] for his overall contribution with [wordbatch][2] and kernels in general .

My favourite positions in a kaggle  competition are 1st and 4th. 1st you get most points/money. 4th, you dont get as many points, but you dont have to reproduce your solution :)

Our validation schema was as follows: 

We used days 7,8 for training and we were making predictions for 9th day (hours [4,14])
For test predictions, we were training on all 7,8,9 and making predictions for the test day (10th)

Things that work for us apart from what it is already in forums/public kernels:

1) [Restacking][3] - When we first started doing Stacking, we could barely get 1,2 points our of it (like from 0.9818 to 0.9820). After adding ALL the features used in our standard modelling to the predictions of the ninth day - we got another +3 boost (to 0.9823). We ended up having around 50 models - mostly lightgbms, but also nns , FMs and some linear models. NNs were on par with LGB models. 

2) We got another +4 from creating WoE ([Weights of Evidence][4]) features for many combinations of all variables (ip,app,device,os)  . This is very strange , because we tried standard target encoding and **it did not work**. This is very strange, because the ordering of likelihood features and woe should be the same/similar (only the range in woe is more condensed) . We are not sure why this happens, maybe it has to do with the binning of Lightgbm?

3)  We got a boost of +4 via sorting ties of time in the same groups of ip,app,device,os, making certain the is_attributed==1 **is always last** [as it was pointed out in the forums][5]. 

Each one of the team members had different features - mine were more related with time-series. Like counts of previous (and next ) days, hours, minutes and seconds of ips,apps,device (and their combinations) . 

Other than that features that measure time between next/previous clicks for various sortings (like ip and app or ip,app,device and os)  were also important. 

The most important feature (importance-wise)  was by far the app (treated as categorical). You could see that certain apps had very different (HIGH) probabilities of **is_attributed**  - I called them for fun tr-APPS!

  [1]: https://www.kaggle.com/anttip
  [2]: https://github.com/anttttti/Wordbatch
  [3]: https://github.com/kaz-Anova/StackNet#restacking-mode
  [4]: https://github.com/h2oai/h2o-meetups/blob/master/2017_11_29_Feature_Engineering/Feature%20Engineering.pdf
  [5]: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/55677
