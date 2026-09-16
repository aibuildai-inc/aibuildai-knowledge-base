# 14th Place Solution with best single model only scored 0.461/0.463 on LB

Competition: commonlitreadabilityprize
Rank: #14
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258209

We will try to be as brief as possible.

**Step 1 - Training Part:**
We have tried all most all the models from Hugging Face, among them, Xlnet for us is the winner.
a. With original dataset. Our best single model is xlnet large model(lb 0.461,private lb 0.463).

b. Using pseudo labels for training. We blend some of our selected models to get a public LB score of 0.455. Using the blended models to generate pseudo labels for the legally public data as external data and continue training. Our best pseudo label model is Roberta large model(lb 0.460,private lb 0.456).

c. A special remark of the training process, We used std as the second loss. e.g.
loss = torch.sqrt(loss_fn(logits, labels.view(-1)))+0.6*torch.sqrt(loss_fn(logits, (labels-stds).view(-1))).

**Step 2 - Avoiding shake up with multi-level stacking.**
Since we have used at the end more than 20 models: Half of them trained with the original dataset, half of them with pseudo labeled dataset. The cv scores are much different. The blend helps us only to reach 0.451 and 0.453 on the public and private LB respectively. How to move on?

Stacking. But so many models with much different cv scores, normal stacking works not so good at this point. We applied multi-level stacking. We used Lasso as the base model for stacking. All the low-score models are placed at level 1, pseudo label models at level 2, linear models at level 3. This setup helps us to reach 14th place.

However, it is not our best score. We selected 12 models and applied 12 Level stackings, i.e. each level just append one model oof/prediction, public and private LB are both 0.448, there is no overfitting. If more models are added, it might be scored higher.

**Step 3 - detailed statistical data analysis**

We worked hard on selecting the final submissions since we noticed there might be a big chance to have a shake-up on the LB. It took us almost 3 days. We did hundreds of plottings and different statistical analyses. It worthed, especially with such a small dataset.

That is it, thanks for reading guys!
