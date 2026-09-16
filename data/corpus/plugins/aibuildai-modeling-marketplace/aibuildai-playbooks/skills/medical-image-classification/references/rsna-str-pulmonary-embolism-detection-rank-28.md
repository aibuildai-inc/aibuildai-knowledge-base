# 28th Place - Quick writeup - Improving the baseline

Competition: rsna-str-pulmonary-embolism-detection
Rank: #28
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193408

I want to thank my amazing team @aerdem4 @drhabib @cateek and @proletheus. This was a very challenging competition and I had a great time collaborating as a team and discussing ideas. Also thanks to the host and winners! I trust the top solutions will positively benefit physicians and those impacted by pulmonary embolism.

Even though our best submission was based off the from the "baseline" kernel that was released a week before the deadline, it's a little bittersweet. It's disappointing that such a high scoring kernel was released so late- I know it frustrated many teams (ours included). Our team was working on other end-to-end and 3D based models. I took on the role of improving the baseline as our backup plan.

This turned into more of an engineering challenge, as we had to balance the limited inference time with such a large amount of data- without exceeding GPU or local memory. It was also challenging because inference time seemed to vary randomly. A submission might complete in 7 hours but then a nearly identical submission would go over 9 hours and fail.

Differences between our solution and the public kernel:
- Trained a b6 in place of b0 for the stage 1 models.
- Changed the inference loop to bag predictions from all 10 (5 folds x 2) stage 1 models.
- Bagged 5x predictions from stage 2.
- Changed the code so that it would only predict the private test during inference and used offline calculated public test predictions.
- Modified the code to only loop through the dataloader once instead of twice for stage 1 predictions.
- Ahmet used some magic to tweak the stage 2 model to get some added boost, including modifying the loss function.

What didn't work:
- ResNext101 in stage 1 models, although it had a slightly better CV score.
- Parallel GRU + LSTM for stage 2 model.

Our solution also meets the required criteria and shouldn't have conflicting label predictions. Our best submission ignoring these restrictions would've given us a private LB score of 0.177 but we didn't select it to stay within the rules.

I'd also like to personally thank Z by HP and NVIDIA for providing me the Z8 desktop which I put to good use on this challenge.
