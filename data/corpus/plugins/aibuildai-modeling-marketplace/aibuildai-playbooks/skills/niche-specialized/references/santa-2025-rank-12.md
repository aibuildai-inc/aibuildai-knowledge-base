# Santa25 12th place

Competition: santa-2025
Rank: #12
Source: https://www.kaggle.com/c/santa-2025/writeups/santa25-12th-place

# Santa25 12th place

## Introduction

First of all, our team would like to thank the organizers for making this competition possible. Also, we want to express our gratitude to the participants who shared their original ideas, findings and codes during the competition. This was the competition with over 3300 teams and we are very pleased to have finished in 12th place.

The key elements of our final result:

- Efficient codes for optimization using [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) and [memetic algorithm](https://en.wikipedia.org/wiki/Memetic_algorithm)
- The active use of [Spyrrow](https://github.com/PaulDL-RS/spyrrow)/[Sparrow](https://github.com/JeroenGar/sparrow) code
- An approach to construct initial guesses for large Ns using Spyrrow, which worked quite well. In this approach, the tree lattice is replaced with a concave hull, which is then treated as a single polygon during the Spyrrow optimization. For more information, refer to the **Details** section.
- An approach to avoid current local minima using the aspect ratio perturbation method. Specifically, we started with the current square solution, which has an aspect ratio of 1. Then, we optimized this solution to aspect ratio of 0.9-1.1 and back to aspect ratio of 1. We used a memetic algorithm for these steps, which was computationally efficient for this purpose. After such perturbation, simulated annealing was applied. This approach worked quite well throughout the competition.
- Transfer of well-scoring tree layouts to their neighbours. For example, the construction of initial guess for the (N-1)-tree solution via removal of a tree from the good N-tree solution. The removed tree might be a border tree or one with a large space buffer around it. Similarly, an initial guess for the (N+1)-tree solution can be constructed by adding a tree to a good N-tree solution. The optimal location for adding the tree can be determined by examining the resulting score of the (N+1)-tree configuration.

## Details

### Solutions for small N

For small Ns, we simply used Spyrow to pre-optimize the solution for multiple seeds. Then, for solutions with good scores, we applied simulated annealing. The strip height requested by Spyrrow was gradually decreased in accordance with the score currently achieved for the considered N.

### Solutions for large N

Many of our solutions for large Ns consist of a large lattice part and bands of free trees on one or two connected sides. The figure below provides examples of such solutions for N = 128, 132, 181 and 192.

We found an efficient way of constructing initial guesses for such solutions using Spyrrow. First, the lattice part of required size is constructed. Secondly, the lattice part is replaced with a concave hull representation, enabling it to be treated as a single object in Spyrrow optimization. Spyrrow is then used to place free trees near/around the constructed convex hull representation. Finally, the concave hull is replaced back by lattice trees and resulting tree configuration serves as an initial guess for subsequent optimization using simulated annealing. The figure below illustrates this process. 

Spyrrow optimizations required some experimentation with the requested strip height, as well as trying multiple seeds. The parameters of the tree lattice used in these solutions were determined by minimising the product of two perpendicular translation vectors. This optimization yields the densest possible configuration of an infinite tree lattice with perpendicular translation vectors.

Another type of solutions that achieved good scores for large Ns also consists of a tree lattice block in the central region. Non-lattice trees are located near the corners and along one one or two opposite sides. The figure below provides examples of such solutions for N = 88, 115, 159 and 170.

For this type of solutions, we did not have precise strategy to prepare initial guesses. Such solutions emerged during the optimisation of certain Ns, and were then transferred to neighbouring N-1 and N+1 solutions using the strategies briefly described in the introduction. The aspect ratio perturbation method described in the introduction was particularly effective for this type of solutions.

Finally, we can look at our two highest-scoring solutions, both of which have a score of 0.328x but correspond to different tree layouts.

