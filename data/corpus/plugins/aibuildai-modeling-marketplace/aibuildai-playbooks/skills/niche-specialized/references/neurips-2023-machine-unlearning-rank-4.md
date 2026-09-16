# Unlearning solution (3rd rank)

Competition: neurips-2023-machine-unlearning
Rank: #4
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/459334

# 1. Presentation and general context

Hello! I am Seifeddine Achour an engineering student specializing in mathematical modeling and data science at ENIT, enrolled simultaneously in a Master's program in image processing and living complexity of Paris cité. Also, I am the co-founder of Google's machine learning community MLAct.

I conducted many AI researches, some of them were academic, others were professional, but most of them were brought by me based on spontaneous questions that came to my mind, on which I work to find them a solution just out of passion and interest. One of these questions was: What if I have a large model and I want to let it forget some data while keeping almost the same performance on the other data without retraining it from scratch, as it may take much time?
I mulled over that for some time until I stumbled upon this competition, which allowed me to focus more on it, and then authored a scientific paper on this topic.


# 2. Overview
The competition aims to find the model state that maximizes the memory on the wanted data and minimizes it on the unwanted data. In my article, I used a different approach which worked perfectly in the Starting Kit using the CIFAR10 dataset and Resnet18 model as shown in the following figure:



However, in this competition, I encountered some compiling problems that were not understandable of course as I don't have access to the error interface.
Therefore, I used a different approach starting from our main problem which is The heavy computation time related to retraining from scratch, which is based on 2 main axes:

1. Instead of running a forgetting algorithm which is time-consuming, initially, the algorithm confuses instantly the model’s vision by changing a bit the state of the convolutional weights while keeping in average the original information and then training it on the retainset.

2. Due to the classes’ unbalance in this dataset specifically, I introduced a new criterion that gives more importance to the majority class “Class0” to minimize the outlier decisions.




# 3. Approach details
As mentioned in the previous section, the problem of retraining from scratch is The heavy computation time. The main reason behind this heaviness is related to the slow backpropagation of the model state using gradient descent on all parameters starting from an initial random state. That's why, to surpass this problem the approach consists of retaining on average the global information existing in the original trained model and just confusing a little bit its vision to reconstitute it later properly. This will be achieved by deviating a bit the convolutional parameters' from its actual state while keeping on average the old parameters, then reinforcing the model's knowledge on the desired data through epochs. 

Let’s understand the maths behind that, we consider, initially,  w_1...w_n the weights of the convolutional layers as random variables that follow the Gaussian Law 𝓝(m_i ,𝜎^2) , where mi is the original weight and 𝜎=0.6. After that, we continue the training process on the retaining set for 4 epochs using the SGD algorithm, learning rate 0.0007, momentum 0.9, and the strong convexity component weight_decay=5e-4. Before starting the final training epoch we apply a light variation in the weights to avoid the non-differential minima in the loss landscape then we finalize the last epoch.

The second crucial component to consider while unlearning the model is the class imbalance present in the dataset as mentioned in the organizers’ article and shown in the following figure:


[1]

To highlight this fact while training, I introduced a customized loss function which is based on the cross entropy principle but additionally, it gives weight = 1 to class 0 and 0.05 to the rest of classes. The loss is represented as follows :




# 4. Results
The scoring system in this competition is based on the degree of similarity between the retrained model and the unlearned model. The highest score that my model obtained using the first 50% of the test set was 0.0968 and the average of all scores of this same algorithm was 0.092. Using the second 50% of the test set, it obtained 0.0885 as a final validation score. 

It is also important to note that the whole algorithm duration is around 6 hours from the confusing process to the training through 4 epochs.


# 5. Discussion
In this section, I would like to highlight three aspects:

1. The increase in the number of epochs seems to be unuseful above a certain threshold. Although it may be challenging to confirm it without access to the output interface to make a proper analysis and interpretation of model performance, I assume it is due to an overfitting which happened because of the class unbalance.


2. Despite the relative fastness of the algorithm that I implemented in the competition and its good score using only the retain set, the use of the forget set can be beneficial as it adds more information to help the model make the right decision. I confirmed that using the approach that I developed theoretically in my article on a regression problem, and used it to conduct experiments on classification problem using the starting kit of this competition.


3. It sounds logical that the retrained model will have the best forgetting performance on the forget set while keeping the exact same accuracy on the retain set. Otherwise, according to the experiments that I have conducted, If we are satisfied with a lightly low retain accuracy  we can get much better results in terms of forgetting performance which can be more beneficial in many applications as we can interpret from the following figure:




# 6. References
[1]  NeurIPS Machine Unlearning Competition organizers, (August 2023), “Evaluation for the NeurIPS Machine Unlearning Competition”

Code link: https: //www.kaggle.com/code/seifachour12/unlearning-solution-4th-rank
