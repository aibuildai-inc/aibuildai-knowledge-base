# Silver Medal Solution (15th)

Competition: santa-2023
Rank: #14
Source: https://www.kaggle.com/c/santa-2023/discussion/472437

This year too, we won a silver medal in the top group. You may want to read the already published [#1 solution](https://www.kaggle.com/competitions/santa-2023/Discussion/472405) and solutions from other winners. 
But please allow me to make my [code](https://github.com/k-harada/santa2023) public and write the solution, to prove that we are not cheating. 


### Cubes: 
We can solve regular patterns by using [nxnxn solver](https://github.com/dwalton76/rubiks-cube-NxNxN-solver), but the way of counting moves is different between this solver and the competition, so if you use it as is, It will be less efficient.  For example, in the worst case, 16Uw2 is 1 move for the solver (and for humans), but it is counted as 32 moves in this competition.  

We first aligned the edges and corners by repeatedly using the solver on a 5x5x5 problem. The approach to extracting 3x3x3 subcubes with corners and centers is shown in the first solution. By repeating 5x5x5, we can solve all the diagonal parts ((1,1) of 4x4x4), the cross parts ((1,2) of 5x5x5), and the edge parts ((0,1) of 4x4x4). 
All that is left to do is align the inside ((1,2) of 6x6x6). We did greedy search to find efficient next 3-rots. (We called it "the magic of swapping three points" :-) ).  
Using a 5x5 solver is also not efficient, but we didn't have an efficient way to align the edges, so we did it this way. 

For the cube N0;N1;.., by coloring it and performing the same method, the edges and corners will be aligned.The rest is essentially the same.  


### Globes: 
I solved it by reducing m x n to multiple 1 x n. Since using f_k freely will affect other layers, I limited it to f_0 and f_n.  
In 1 x n globes, there is also "the magic of swapping 3 points", so we can get a solution with sufficient performance.

### Wreaths: 
I was able to get a good enough solution using a simple heuristic.


Post-processing by @tomokiyoshida and parallel execution with different parameters by @sfujiwara significantly improved the score.
