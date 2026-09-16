# Summary of rank 6 solution

Competition: intel-mobileodt-cervical-cancer-screening
Rank: #6
Source: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/35089

Congratulations for the winners for a great effort!  Glad to see the top solutions differ significantly in score, look forward to hearing your solutions！

The most significant gain for my solution is using SSD to create bounding boxes for the Os.  Huge thanks to [Paul][1] for providing the bounding box annotations.

My model is an ensemble of vgg-19, xception, resnet 50 and inception v3 models.  The top two models are: vgg-19 fine-tuned with 7 frozen layers and xception with 2 frozen layers.  They are similar in score.  The rest, various fine-tuned and bottle-necked models, are significantly weaker but added slightly to the ensemble's effectiveness.  All models ran with 3-fold CV, and are combined with linear regression to produce the final result.

I attempted to include some "leak" features in a second submission, but its result is much worse.  As far as I can see, the final results are straight.  Great thanks to the organizers and admins for making this a useful competition!

The whole pipeline ran on a gtx970 for five days, barely making the submission deadline.  The top two models ran for 1.5 and 2 days each.

  [1]: https://www.kaggle.com/deveaup
