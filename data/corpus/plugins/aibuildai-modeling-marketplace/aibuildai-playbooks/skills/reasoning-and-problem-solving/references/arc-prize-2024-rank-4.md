# 4th place solution

Competition: arc-prize-2024
Rank: #4
Source: https://www.kaggle.com/c/arc-prize-2024/discussion/550414

First and foremost, I would like to extend my utmost gratitude to Kaggle and the ARC Prize organizers for hosting this remarkable competition, which represents a pivotal step in raising awareness and advancing efforts toward addressing the challenges of Artificial General Intelligence (AGI).

The code is linked here: https://www.kaggle.com/code/williamwu88/fork-of-small-sample-arc24-7d97ca

- The notebook is essentially using classical techniques such as DSL (Domain Specific Language), decision tree, CNNs, and builds upon winning solutions from the 2020 competition, refining them with updates & increased computational power introduced in 2024. These changes enhance algorithmic efficiency and adaptability, leading to improved score with ensembling techniques.
 
- The notebook integrates insights from top solutions of the 2020 competition, such as the DSL, which has proven effective for solving multiple tasks. The referenced work by participants like @icecuber, @golubev, @szabo7zoltan, @ilialar, @mehrankazeminia, and @somayyehgholami was adapted with modifications as needed. We also referred to Michael Hodel and team’s public GitHub repository on ARC-DSL (https://github.com/michaelhodel/arc-dsl).

- A key factor that increased the score from 2020 solutions in 2024 were the increased computational power of Kaggle Kernels (increased to 30 GB RAM), as well as the time allowed (increased from 9 hours to 12 hours). This allowed us to run the algorithms at greater search depth, for longer epochs, as well as fit in more models into the ensembles.

A list of machine learning techniques (roughly in order of importance) used:
- DSL (domain-specific language) along with search over a directed acyclic graph (DAG)
- Decision Trees
- Convolutional neural networks (CNNs)
- Data Augmentation and Preprocessing (such as diagonal flips, color switching, using symmetry, etc.)

For ensembling, key techniques used were majority vote, custom logic (for example, icecuber solutions were manually prioritized and exempt from majority vote), and probabilistic trials (where the probability of choosing a model is approximately proportional to the number of new problems it solves).

Overall, I did not expect to be in the prize position (originally was 7th position, went to 4th after some teams withdrew) using the classical machine learning techniques. In conclusion, after reading the solutions, I strongly believe that the Test-Time Training (TTT) approaches by the top teams are currently the state-of-the-art and the most promising approach. Time will tell whether TTT is sufficient to crack the ARC prize!

Thank you to Kaggle, and the ARC Prize organizers once again!
