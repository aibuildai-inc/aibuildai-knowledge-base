# 2nd place solution

Competition: neurips-2023-machine-unlearning
Rank: #2
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/458721

At first, thanks to Kaggle for hosting such an interesting competition and every Kagglers. It’s been a truly memorable experience for me, not only because of the active discussion zone, which provides interesting and useful directions for thought and practice, but also because of the meaningful topic of unlearning. I'm surprised to have this chance to introduce my solution here.
My solution is based on a 2-stage training method, which contains a one epoch forgetting stage on forget dataset and an 8-epoch adversarial fine-tuning stage on both forget dataset and retain dataset. Here are the details:
## Overview:
(1). In order to let the model initially forget in the logits space, we optimize the KL-divergence between the output logits and a uniform pseudo label.
(2). An adversarial fine-tuning stage:  
**Forget Round:**
     Instance-scale SCL(self supervised contrastive learning) could be a beneficial method when facing some unsupervised learning task such as clustering feature embeddings. Self supervised contrastive learning aims to pull the distance between positive samples and their enhanced samples closer, and to push the distance between positive samples and all samples further. In my assumption, we need to let the forget sample as uniform as possible in feature space, which could be seen as pushing the distance between any forget sample and all retain samples further. Hence, I have proposed two kinds of contrastive loss here:
1.	For a forget sample x, use its enhanced version x’ as a positive pair, and use all samples in the retain batch as a negative pair. The final loss formula could be written as this: 


2.	Each retain sample could have the same probability to be chosen as positive sample with the any forget sample, which means they have the same weight in the formula of the contrastive loss. The final loss formula could be written as this:

I have use second loss function as the final submission version (Because the first loss function performs not as good as the second preliminarily, I give up this idea due to the days limitation). I set temperature coefficient t=1.15 here.
the forget training round code here:
```
for sample_forget, sample_retain in zip(forget_loader, retain_ld4fgt):
    t = 1.15 ##temperature coefficient
    inputs_forget,inputs_retain = sample_forget["image"],sample_retain['image']
    inputs_forget, inputs_retain = inputs_forget.to(DEVICE), inputs_retain.to(DEVICE)
    optimizer_forget.zero_grad()
    outputs_forget,outputs_retain = net(inputs_forget),net(inputs_retain).detach()
    loss = (-1 * nn.LogSoftmax(dim=-1)(outputs_forget @ outputs_retain.T/t)).mean() ##weighted CL 
    ##Loss function
    loss.backward()
    optimizer_forget.step()
```
Due to the different steps in forget_loader and retain_loader (here referred to as retain_ld4fgt, a new dataloader for Contrastive Learning), there is an unexpected random shuffle at different checkpoints, which can be viewed as an ensembling method.
**Retain Round:**
      To enhance performance on the retain dataset, fine-tuning it is unavoidable. Here, I just simply retrain the model using cross-entropy loss.

##Some useful tricks:
1. Actually, because of our competition time limitation, the max epoch could be maximized to 6 when retain batchsize is 64. (This conclusion can be continued to use even for adversarial training because the forget round takes little time compared with the retain round).  "I've tried to add more epochs by increasing the batch size of the retain dataloader and the learning rate. Results show that eight epochs of training with a batch size of 256 performs best on the public leaderboard.

2.CosineAnnealingLR could also be useful in forget rounds (Simply adding this, score on public lb increases from 0.084 to 0.091)

##Some directions I have not investigated deeply :
1. Add more forget rounds in an epoch. For example, to intensify the adversarial stage (as a fiercer confrontation may enhance forgetting performance), adding another forget round could increase the total steps of forgetting.
2. Replace the first stage with Contrastive learning loss.

## Conclusion:
During the competition, my thinking went through multiple stages, and each stage had a different understanding of the competition and tasks. In fact, many excellent codes or discussions in the forum have been of great help to me. For example, @Maria Gorinova 's [evaluation notebook](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/454949) is an excellent work that help you quickly understand the evaluation metrics. I have also witnessed many vastly different ideas on the forum. I believe that each idea has its own reasons, but the competition scenes are limited and idealized. While high-scoring methods may have limitations in practical scenarios, their underlying innovative approaches are universally applicable. I always look forward to seeing any creative solution.

Solution link:[2nd place Solution](https://www.kaggle.com/code/fanchuan/2nd-place-machine-unlearning-solution?kernelSessionId=153137657)
