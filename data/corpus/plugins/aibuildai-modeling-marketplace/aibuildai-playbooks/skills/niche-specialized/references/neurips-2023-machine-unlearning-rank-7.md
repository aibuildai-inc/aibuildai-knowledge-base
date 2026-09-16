# 6th Place Solution for the NeurIPS 2023 - Machine Unlearning Competition

Competition: neurips-2023-machine-unlearning
Rank: #7
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/458740

First of all, we would like to thank the organizing committee, it was an incredible experience, and we are so grateful for all the hard work and effort you and your team put into making it happen.

#Context
- Business context: https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/overview
- Data context: https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/data

#Overview of the approach
In our solution, we present a simple yet effective machine unlearning approach that uses a selective model parameter reset, a warm-up phase employing knowledge distillation and finally, a fine-tuning phase. Specifically, our method focuses on resetting the first and last layer of the original model, which is trained on the full training set. By resetting the first layer, we enable a disruption on the initial hidden representations of the data, leading to a direct decline in model performance, facilitation the forgetting of previously trained data. Resetting the last layer enables the model to diverge from its original output patterns, allowing it to approximate a new output distribution, which can result in similar output distribution to that of a model trained on a specific data subset, i.e., the retain set. 
- We followed the data processing pipeline as provided in the starter notebook(run-unlearn-finetune), without further data augmentations.
- We submitted two final versions: one that uses the entire validation set and another that uses only samples from the first two classes for the warm-up phase. Note that these two classes were the only ones with samples in the forget set.
- Our method does not require the forget set.

#Details of the submission

## Insights from Cifar-10
The initial phases involve resetting the first and layers of the original model. These two layers are chosen because the first layer significantly influences the rest of the model layers and the last layer determines the model’s final output distribution. With this reset step, we enable the model to deviate from its original state. 
In addition, empirical observations from local experiments on the CIFAR-10 dataset showed that these two layers exhibited mostly negative cosine similarity between model weights trained on the full training set and models trained from scratch on a smaller subset (i.e., the retain set). 
Thus, in terms of global weight parameters, this means that the two models found a similar solution with respect to predictive performance, but with different directions (See Figure "Weight Similarities" on the CIFAR-10 dataset between the pretrained and retrained on retain set model).

### Weight Similarities 


Furthermore, we measured the differences in terms of activation distributions between the pre-trained model and the model trained from scratch on the retain set. This experiment revealed that the differences in terms of cosine similarity are clearly evident even regarding their activations (See figures below) These observations lay the foundations for resetting the first and the last layer parameters of the original model. 

### Similarity of Activations after MaxPooling layer


### Embedding Similarity (after AvgPool layer)

Finally, we performed experiments with baseline unlearning algorithms including simple finetuning, finetuning by resetting only the final layer, eu-k and cf-k forgetting [1], unlearning by maximizing the loss on the forget set [2], unlearning with bad teaching [3] and SCRUB [4], which minimizes the divergence of a student model from the teacher on the retain set and maximizes the divergence on forget set. In all of the above cases, none of the methods achieved significantly higher public score compared to the baseline fine-tuning approach.

After resetting the first and the last layer of the network, we include a rapid warm-up phase using the validation set. The goal in this phase is to prepare the model for the third phase by minimizing the Kullback-Leibler (KL) divergence between the outputs of the original pre-trained model (teacher) and the reinitialized model (student).

The final phase includes fine-tuning the student model on the retain set. In this stage, we combine three types of losses: standard cross-entropy loss, soft cross-entropy loss and KL divergence. The cross-entropy loss ensures the model's accuracy on the underlying task using hard labels from the retain set. The soft cross-entropy loss involves comparing softened predictions of the student model with soft labels from the teacher model.  The KL divergence loss combined with the soft cross-entropy facilitates rapid knowledge transfer and broader information capture. Additionally, the cosine annealing scheduler is integrated to optimize training dynamics, promoting faster convergence and efficient learning.
The Figure below shows a summary of the proposed pipeline.

## Unlearning Pipeline Visualization


# Final Submissions
As mentioned earlier, we selected two variations of our pipeline. The first uses the entire validation set and the second employs the warm-up phase only a subset, specifically using the samples from the first two classes. This inspiration of this strategy derives from the distribution of forget set, which exclusively contains samples from the first two classes (0,1).

To extract the samples belonging to the first two classes, we used the following function:

```python
def get_val_loader(batch_size):
    val_ds = HiddenDataset(split='validation')
    samples = []
    for sample in val_ds:
        y = sample["age_group"]
        if int(y) == 0 or int(y) == 1:
            samples.append(sample)
    val_loader = DataLoader(samples, batch_size=batch_size, shuffle=True)
    return val_loader
```

The scores of these two approaches are summarized in the following table:

| Warm-up Approach                     | Public Score | Private Score |
|------------------------------|--------------|---------------|
| All classes              | 0.08383      | 0.07219       |
| Classes 0,1 | 0.08324      | **0.07831**      |

Notably, both approaches resulted in almost equivalent public scores. However, the approach which uses the first two classes from the validation set for warming-up the model demonstrated  improved performance in the hidden test set.  

##Useful Tricks
- Resetting a portion of model parameters 
- CosineAnnealingLR Scheduling for Finetuning Phase

##Strategies
- Threefold Optimization Loss during Finetuning Phase 
- High Temperature in both stages

## What didn't work
- Use of class weights (Despite their use in the original training pipeline)
- Algorithms that did not employ resetting resulted lower scores
- Variants of retraining (Starting from pretrained model on ImageNet resulted to low scores or threw exception probably due to exploding gradients)

#Conclusion
In conclusion, throughout this competition we gained a better understanding of Machine Unlearning which has been both enlightening and challenging. Special thanks also to the non-Kaggle participant of our team, Nikos Komodakis for his contribution.

#Sources
1. Goel, S., Prabhu, A., Sanyal, A., Lim, S. N., Torr, P., & Kumaraguru, P. (2022). Towards adversarial evaluations for inexact machine unlearning. arXiv preprint arXiv:2201.06640.
2. Golatkar, A., Achille, A., & Soatto, S. (2020). Eternal sunshine of the spotless net: Selective forgetting in deep networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9304-9312).
3. Can bad teaching induce forgetting? Unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7210-7217).
4. Kurmanji, M., Triantafillou, P., & Triantafillou, E. (2023). Towards Unbounded Machine Unlearning. arXiv preprint arXiv:2302.09880.
- Starter Notebook (run-unlearne-finetune) https://www.kaggle.com/code/eleni30fillou/run-unlearn-finetune


Code for [7th Place Solution](https://www.kaggle.com/code/stathiskaripidis/unlearning-by-resetting-layers-7th-on-private-lb)

## Team
- [Efstathios Karypidis](https://scholar.google.com/citations?user=jif2JYsAAAAJ&hl=en), PhD Student Archimedes Unit, Athena RC | NTUA
- [Vasileios Perifanis](https://scholar.google.com/citations?user=V1t6u_YAAAAJ&hl=en&oi=ao), PhD Student, DUTH
- [Christos Chrysanthos Nikolaidis](https://scholar.google.com/citations?user=srkOEMYAAAAJ&hl=el&oi=ao), PhD Student, DUTH
- [Nikos Komodakis](https://scholar.google.com/citations?user=xCPoT4EAAAAJ&hl=en&oi=ao), Assistant Professor, (non-Kaggle Participant), UoC
- [Pavlos Efraimidis](https://scholar.google.com/citations?user=Dxz0pLsAAAAJ&hl=en), Professor, DUTH
