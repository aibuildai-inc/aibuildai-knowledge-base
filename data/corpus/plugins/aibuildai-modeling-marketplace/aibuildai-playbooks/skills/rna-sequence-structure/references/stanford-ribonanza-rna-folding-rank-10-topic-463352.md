# 10th Place Solution for the Stanford Ribonanza RNA Folding Competition

Competition: stanford-ribonanza-rna-folding
Rank: #10
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/463352

Initially, I had not intended to publish my solution since I used an interesting method for the loss (that I did not see in other solutions, although I might have missed it), and since I did not finish within the price range, I wanted to keep it to myself for some other future compatible competition. However, since apparently there are plans for a paper by the hosts, I eventually opted to publish it.

# Context section
[Business context](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding).
[Data context](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data).

# Overview of the Approach
I used a 1Dconv+transformer model. The model does not include positional encoding since the conv layers take care of positional relationship ‘automatically,’ allowing for straightforward generalization for longer sequences. I based my model on a previous work of @hoyso48 in the ISLR competition; see [HERE](https://www.kaggle.com/competitions/asl-signs/discussion/406684) and [HERE](https://www.kaggle.com/competitions/asl-signs/discussion/406978). BTW, congrats for 2nd place. You know, Hoyeol is probably the person from whose work I learned the most. This solution is already the second gold I get, thanks to the knowledge I got from his work. So, this is a special thank you. 
Hoiso’s original model was causal in time; I changed it to be symmetric to both sides by employing suitable paddings and maskings. 
I trained two kinds of models; model_1 was trained only on the sequences, and model_2 included CapR and bpp sum&max per nucleotide.
Things that helped me (I did not perform a proper ablation study, so I will not give exact numbers most of the time):
1. Clipping the predictions with a sigmoid (regular clipping also helps, but a sigmoid helps even more). It also made the convergence smoother.
2. One-hot encoding of the nucleotides instead of embeddings (this is mainly for the seq-only model. When including more features, one-hot is a given)
3. CapR
4. BPP sum&max
I also found the injection of bpps values to the attention helpful, even more so with 2D conv layers on the bpps in various schemes. HOWEVER...I was very very suspicious of bpps. I feared that they would be much worse on the private test set since it consists of new sequences, and I thought that even if it was good for the public set, maybe the BPPs models were developed on similar sequences, but for the private set, it would fail. So I chose only to include sum&max per nucleotide, thinking that these ‘averaged’ values would be less prone to overfitting. Also, tbh, I did not have enough time at the end to do proper research and ensemble of the more complicated models due to trying too many ideas in too little time, lol.
5. Weighted loss function:
Let's start from the end.
loss(reactivity, reactivity_error, pred)=MAE(reactivity, pred)\*potentials  
with:  
potentials = (1-2\*dist.cdf(reactivity-|reactivity-pred|)  
and:  
dist = normal_distribution(mu=reactivity, sigma= reactivity_error)
Now, to explain why.
I thought a lot about the loss function. Since we were given the errors, I searched for an appropriate way to incorporate uncertainty in the measurements into the loss. I was a bit surprised when I could not find something useful- of course, I maybe did not dig deep enough, but at the very least, it seems that a lot of digging is required. I found several things about incorporating uncertainties into the predictions (i.e., predicting the uncertainties of the predictions), but I wanted the opposite.
Now, the obvious thing to do is directly weigh the loss by the error- to give some examples, Hoiso (2nd place) used log1p error, and I saw suggestions like 1/(1+err), etc. However, I needed more than this: it punished high and low errors too much. To understand why, let's talk for a second about the meaning of error- generally, it means that instead of knowing the exact value, we can only give an approximation for the true value, with a distribution of probability for it to be any number, and the probability given by normal distribution with sigma = error. When we say that reactivity=0 ± 1, we give a probability of 0.68 for the exact value to be between -1 to 1 (i.e., within one standard deviation). 
Lets assume that we have reactivity1 =0 ± 1 , reactivity2 =0 ± 3 , pred1 =pred2 =100 . If we normalize, for example, by 1/(1+err), and with MAE, then loss1 =50, loss2 = 25. So the first predictions contribute to the loss twice as much as the second one, even though, if we look at the distribution of probabilities- the first one has 0.68 to be in [-1,1] and the second one has 0.68 to be in [-3,3] (and 0.95 to be in [-6,6]) so for a prediction so far away from most of the distribution, I expect both predictions to contribute to the loss about the same. Of course, we clip the predictions between 0,1, making the analysis more complicated and depending on the exact loss and clipping each person applies. 
On the other hand, assume we have reactivity1=0 ± 0.01, rectivity2=0 ± 0.1, pred1=pred2=0.05, then we get loss1=0.0495, loss2=0.0454. This time, they are almost the same- even though we know that reactivity1 has a probability of 0.68 to be in [-0.01, 0.01], which is far from the prediction relatively to the prediction for reactivity2, which lies inside one standard deviation of the ground truth distribution, so I expect it to contribute much less to the loss. 
One way to address this issue is to perturb the reactivity values according to their probability distribution. However, this leads to unstabilized training since many values have a high error, and moreover, they are perturbed outside the range of [0,1], leading to more complications.
I wanted to properly incorporate our knowledge about the probability distribution of the values into the loss function. At this point, I took some inspiration from the shell theorem (check it [on Wikipedia](https://en.wikipedia.org/wiki/Shell_theorem)), and by considering the probability distribution as a form of ‘potential,’ I finished with the above formula. 
This loss helped my convergence. I don’t remember precisely how much, maybe about 0.001? Not by a huge amount. But it certainly helped. It also made it even smoother. I think this method allows the inclusion of high error values while squeezing as much information as possible from them. Also, I believe this is better than trying to pseudo-label high-error values since noisy data is still data with real information inside. We just need to squeeze this information out…
But wait, there is more. What if I suspect the errors we were given themselves are inaccurate?
No problem. Just multiply the given errors by some factor, add some constant errors, and assemble many models with randomly chosen factors.
At this point, I remembered the method of label smoothing for logistic regression, and, well, this is basically a form of label smoothing for regression. Once I framed it as label smoothing, it was easy to find similar ideas, e.g. [this paper](https://proceedings.mlr.press/v80/imani18a/imani18a.pdf). Although it seems like they overcomplicated things there a bit.
Note that this works the same for other losses, e.g., for MSE, multiply MSE by the same 'potentials,' etc.

# Details of the submission
1. Ensembling (obviously):
Here is a nice plot of the power of ensembling for the sequence-only models (model_1)


2. Ensembling models with bpp+CapR with sequences-only models
I trusted my sequences-only models much more than the +bpp&CapR ones, so I submitted one ensemble of pure sequences-only models. I planned the other ensemble to be only +bpp&CapR models. But then, on the very last day, just when I had the two ensembles ready, I looked again at the pictures of the results for R1138v1 predictions. See more details [here](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/444653). Basically, we wanted the models to replicate the following image:

Here is the picture of my sequence-only models ensemble:


 It’s a bit hard to see, but it recover the central line, especially in the 2A3 picture. However, the sidelines are missing. On the other hand, for the +bpp&CapR models ensemble:

Here, it’s the opposite. I have visible traces of the sidelines but almost nothing for the central one (maybe one very vague point).
These images bring an immediate idea...more ensembling…(this was not obvious at first since the BPP&CapR models/ensemble had a much higher validation/public score than the sequences-only ones)
I had only three submissions left. I tried the following weights: 6:2, 5:3, and 4:4 in favor of a sequence-only ensemble. 4:4 had a better score but I was too suspisious of bpp and CapR, so I chose 5:3 weights which had a similar score to the original only bpp+CapR ensemble but was expected to be more robust.
After the competition ended, I tried other weights. Here is the complete analysis:

Even if I chose the best weight, I would still end outside the price range. So I don’t feel too terrible about missing here.  I could end up in 8th place, though, if I chose the 4:4 weights on the last day. Keep in mind that my solution does not include the edge information of the BPPs, as opposed to most (all?) other top solutions. Maybe, except the 3rd place, he got an extremely strong sequences-only solution, much better than mine.
# More things I tried but did not included in the final solution or didn’t work out (possibly also due to lack of time):
Using the external data source as another head for the loss, using the external data source to train a model, predicting on the train set, and use the predictions as features, different predictions for bpps (eterna, contra, etc.) did not contribute too much or at all. Structures (i.e. (.) notation) also were not especially useful. I had plans for the 3D features but not enough time...I tried other schemes for the loss, for example, minimizing log loss (sigmoid) (i.e., framing it as logistic regression since the values are between [0,1]) or directly minimizing the ‘potentials’ (see #5. Weighted loss functions)- did not work. I tried various schemes for direct injection of bpp to the attention with or without 2D conv, and it did help (at least +0.001 in validation, probably would be more if I submitted), but as I said before, I did not trusted the bpps and the public LB enough. 

#Validation scheme
During experimentations, I trained on the middle reactivities of the length 177 sequences (between nucleotides #36 and #116, since we were told that the private set might include data on the nucleotides on the edges). I validated on the length 206 sequences and the start/end of 177 length sequences (up to the #36 nucleotide and from the #116 one). For submission, I trained on sequences from all sizes, including the edges reactivities, and validated each model on a random part of the sequences I did not train on for the specific model.

# Main points to take from my solution
My loss function and ensembling with sequences-only models might have a chance to improve the scores of the top models/ensembles a bit. However, I would not give it too high a chance since their scores are much above mine, and they probably captured most of what there was to capture already. Even so, I think that when we have more data (private test is 1M new sequences, and there are talks about producing ten times more data), bpps and CapR methods would be less and less useful while squeezing more data from the measurements (as I tried to do with my loss function) would become more useful than now. So I still hope that my work will be of some help.

# Sources
I based my model on Hoyso's solution to the ISLR competition; see [HERE](https://www.kaggle.com/competitions/asl-signs/discussion/406684) and [HERE](https://www.kaggle.com/competitions/asl-signs/discussion/406978). 
I used [ratthachat's notebook](https://www.kaggle.com/code/ratthachat/preprocessing-deep-learning-input-from-rna-string) to calculate the CapR values.
Check my [GitHub](https://github.com/shlomoron/Stanford-Ribonanza-RNA-Folding-10th-place-solution) for data preparation and training code.
