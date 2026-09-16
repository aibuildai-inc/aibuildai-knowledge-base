# 9th Place Solution (Forget set-free Approach, 3rd on Public LB)

Competition: neurips-2023-machine-unlearning
Rank: #9
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/458715

# Forget set-free Approach (9th on Private LB, 3rd on Public LB)
First of all, thank you for organizing this meaningful challenge. It provided a great opportunity to deeply engage with the concept of machine unlearning and to exchange ideas with esteemed researchers.

## Context
Business context: https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/overview
Data context: https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/data

## Overview
Our solution consists of *forgetting phase* and *remembering phase*. In forgetting phase, model parameters are stochastically selected and re-initialized. In remembering phase, knowledge preserving loss is calculated between the original model and the target unlearning model to remind the target model about retain set. Furthermore, the forgetting phase and remembering phase are repeated for some cycles to enhance unlearning performance.



Table 1 demonstrates the unlearning performance of our final solutions, compared with other published unlearning methods [1-4]. Note that the public scores are recorded.


## Details
### Experimental Settings
NeurIPS 2023 Machine Unlearning Challenge has specific code requirements. For example, the submissions should run 512 completely independent unlearning algorithms without any pre-compiled code or caching work between each run. Furthermore, these 512 runs should be completed within 8 hours. To obligate these requirements, few cycles of stochastic re-initialization and knowledge preserving process are executed. In detail, the first algorithm has 3 cycles with [1, 2, 2] epochs for each cycle. During the total 5 epochs, cosine learning rate scheduler (init_lr=0.001, T\_max=2) is employed. The second algorithm has 4 cycles with [2, 1, 1, 1] epochs for each cycle and learning rate is set to [0.0005, 0.001, 0.001, 0.001, 0.001] for each epoch. Both algorithms randomly select 6 layers from the selection pool, while allowing replacement. The gaussian noise is sampled from the distribution of zero mean with 0.01 sigma. Note that only retain set is utilized for the experiments.

### Logit Distribution
If the unlearning is successful, the unlearned model should produce similar logit distributions compared with the retrained model. Therefore, in addition to the quantitative metric, we observed the distribution of logits when feeding forget set and retain set to the models. Logit values of forget set and retain set are collected and visualized as a overlapped histograms in Figure 2 and Figure 3. As demonstrated in Figure 2 and Figure 3, Our unlearned models produces closer distributions with the distributions of retrained model, when compared with the naively fine-tuned model. Because the challenge dataset is hidden, MUFAC dataset [5] is utilized to visualize these distributions.



### Explored Methods
Aforementioned unlearning methods come from several interim experiments. In this section, some interim findings are introduced to follow our decision process. 

#### Re-initialization
When stochastic re-initialization is employed in addition to the fine-tuning, the score surpasses that of pure fine-tuning. Furthermore, interestingly, parameter selection using diagonal elements of fisher information matrix decreases the score. These results are demonstrated in Table 2.



#### Data Augmentation
Table 3 describes the effect of data augmentation. Gaussian noise, generated from a gaussian distribution with a mean of 0.0 and standard deviation of 0.1, is added to the retain set images.



#### Loss Functions
Table 4 describes the effect of different loss functions. Between the loss functions, MSE loss outperforms other loss functions, such as cross-entropy loss and L1 loss. 



#### N Cycles
It is unable to improve unlearning performance by simply increasing parameter selection ratio for re-initialization. However, we find that repeating cycles of forgetting phase and remembering phase significantly improves the performance. Table 5 demonstrates these improvements.



#### Layer-wise vs. Element-wise
In our experiments, layer-wise parameter selection highly surpasses element-wise parameter selection method, as shown in Table 6.



#### Selection Pool
Furthermore, it is effective to prevent some layers from the re-initialization. In our experiments, fully-connected layers and projection-shortcut layers should be excluded from parameter selection and re-initialization. Table 7 demonstrates these results.



## Conclusion
Our solution exclusively utilizes the retain set, omitting forget set and validation set. This can be effective when the forget set is already delected or its usage is prohibited.
However, as our solution involves random selection of layers, the unlearning performance may vary depending on the specific layers chosen.

## Sources
[1] Laura Graves, Vineel Nagisetty, and Vijay Ganesh. Amnesiac machine learning. In Proceedings of the AAAI Conference on Artificial Intelligence, May 2021.
[2] Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. Unrolling sgd: Understanding factors influencing machine unlearning. In 2022 IEEE 7th European Symposium on Security and Privacy (EuroS&P), pages 303–319. IEEE, June 2022. ISBN 9781665416146, 9781665416153. doi: 10.1109/EuroSP53844.2022.00027.
[3] Eleni Triantafillou, Fabian Pedregosa, Jamie Hayes, Peter Kairouz, Isabelle Guyon, Meghdad Kurmanji, Gintare Karolina Dziugaite, Peter Triantafillou, Kairan Zhao, Lisheng Sun Hosoya, Julio C. S. Jacques Junior, Vincent Dumoulin, Ioannis Mitliagkas, Sergio Escalera, Jun Wan, Sohier Dane, Maggie Demkin, and Walter Reade. Neurips 2023 - machine unlearning, 2023.URL https://kaggle.com/competitions/neurips-2023-machine-unlearning.
[4] Alexander Warnecke, Lukas Pirch, Christian Wressnegger, and Konrad Rieck. Machine unlearning of features and labels, August 2021.
[5] Dasol Choi and Dongbin Na. Towards machine unlearning benchmarks: Forgetting the personal identities in facial recognition systems, November 2023.

## Supplementary Material
The detailed explanation of our methods can be found in pdf below: https://www.dropbox.com/scl/fi/izmzhmj3ktqk3ze6rhjv6/Kaggle_Unlearning_Challenge_Solution.pdf?rlkey=9pqw47izw4nuuthanw3czu1gi&dl=0

Our solution code can be found in: https://www.kaggle.com/code/jaesinahn/forget-set-free-approach-9th-on-private-lb
