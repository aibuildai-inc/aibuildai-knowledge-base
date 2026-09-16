# 3rd place solution

Competition: neurips-2023-machine-unlearning
Rank: #3
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/459200

First of all, thanks to Kaggle and Google for such an exciting competition. It was a great experience.

### Summary

In this work, we developed a gradient-based re-initialization method. We assumed that if the gradients of the weights in the model, specifically in the retain set and forget set, are similar, it becomes challenging to forget information from the forget set during the retraining of the retain set. To address this, we selectively reinitialized weights with similar gradients between the two sets, followed by retraining the model.

### Method

The proposed gradient-based re-initialization method for unlearning consists of three main steps:
1.	Gradient collection: Gradient information is collected from the forget set and the retain set. The forget set’s gradient is collected using gradient ascent from cross-entropy loss with the forget set label, while the retain set’s gradient is collected using gradient descent cross-entropy loss with the retain set label. Due to the unequal sample sizes between the retain set and forget set, random sampling was used from the retain set to match the number of samples in the forget set for gradient collection. 
2.	Weight initialization: Based on the gradient information collected in the first step, a percentage of the convolution filter weights are re-initialized. The weights with the smallest absolute values of gradient information are re-initialized. (Our best result initialized 30% of the weights) The weights are globally unstructured initialized, following the same He initialization (mode="fan_out", nonlinearity="relu") from torchvision Resnet codes.
3.	Retraining: The model is retrained with the retain set. The learning rate for the Uninitialized weights uses 1/10 of the base learning rates. This is accomplished by scaling the gradient of the Uninitilaized weight by 1/10. 

### learning rate scheduler

The learning rate scheduler is also an important factor. A linear decay learning rate scheduler with a few warmup epochs consistently produces better results than a linear decay learning rate scheduler or a linear increase learning rate scheduler.

### Randomness

Randomness is also an important factor. Selecting weights using a subset of the retain set(resulting in a different initialization for each run) results in better performance than using the entire retain set. 

### Image
- gradient collection

- Weight initialization


### Our [code](https://www.kaggle.com/code/nuod8260/targeted-re-initialization-3rd-on-private-lb)


Affiliation: School of Artificial Intelligence, College of Computer Science, Kookmin University

Doun Lee
Email: Idoun8260@kookmin.ac.kr

Name: Jinwoo Bae
Email: bgw4399@kookmin.ac.kr

Name: Jangho Kim (Professor)
Email: Jangho.kim@kookmin.ac.kr
