# 21st Solution using only Kaggle Kernels

Competition: bengaliai-cv19
Rank: #21
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136056

I would like to thank @seesee a lot, because he was very generous to share his notebooks: https://www.kaggle.com/c/bengaliai-cv19/discussion/134161 I was not going to join this competition because I have no GPU, but in the last week, I have decided to join after seeing these nice kernels. I could only use 2x30 TPU hours for this competition.

The baseline was scoring 0.9708 on the public LB. Here is what I have done on top of that:
* I was aware that **random split** for train and validation does not represent the split for the real test set but I was lazy to change it. I just kept it in mind.
* Every class contributes to the metric equally. Therefore, I have **multiplied the predictions by inverse class frequencies** before getting the maximum. This gave 0.007 boost on Public LB. (around 0.003 boost for the validation set)
* I have changed the backbone model from **Efficientnet B3 to B4**. This gave 0.002 boost.
* I have changed the augmentation from mixup to **cutmix ** using the implementation here: https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu I have only updated it so that it does the cuts as rectangles rather than squares (and fixed a small bug, I believe). Thanks @cdeotte for this nice kernel. cutmix did not improve my LB but improved the local validation.
* I have changed the learning scheme. **Decreasing learning rates gradually** gave me 0.007 boost. I could make use of it more but TPU run-time limit was 3 hours. If I had higher limit, I could end up in gold zone. Or maybe the opposite, having an underfit model gave me a shake-up.
* I have ran the model **on whole data**, rather than 80% train split. This gave 0.001 boost.
* I have **blended the last two runs** with different seeds. This gave me 0.001 boost.
* Considering that my split was not representative and having more benefit from multiplying by inverse class frequencies than expected, I have decided to exploit a bit more by weight = np.power(weight, 1.2). This improved public LB by 0.0006 but seems to improve private even more. Probably because there are more **unseen graphemes** in the private set.

So it was possible to get 21st place using only Kaggle Kernels in a week. Reading discussions and kernels carefully is enough for getting around this position. For top 10, usually more creativity is needed.
