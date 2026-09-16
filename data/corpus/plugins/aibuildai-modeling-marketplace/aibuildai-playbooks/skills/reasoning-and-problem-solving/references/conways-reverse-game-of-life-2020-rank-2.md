# 2nd place solution write-up,Team "Under a penny", ebouteillon part 🔥

Competition: conways-reverse-game-of-life-2020
Rank: #2
Source: https://www.kaggle.com/c/conways-reverse-game-of-life-2020/discussion/200531

First, I would like to thank my teammates @knstqq , @maximim , @robertjohncannell (alphabetical order) for their ideas and work. It was really amazing to work with you guys!


During this competition, I tried a lot of different approaches. Here are those giving best results:


# Genetic algorithm using pytorch 🔥

I looked up existing Kaggle notebooks using Genetic Algorithm, but none of them were using a GPU accelerator (sorry if I am wrong and missed one). So I implemented one.

Implementing this genetic algorithm from scratch using pytorch was hard but I learned a lot in the process about genetic algorithm (first purpose), but also on pytorch and how to optimize using NVIDIA Nsight.

This solution performed very well during this competition as the below notebook alone scores in the top-10 leaderboard. This new approach helped improving final solution of our team "Under a Penny".

Notebook with GA on GPU : [Top 10 with First Genetic Algorithm on GPU! 🔥](https://www.kaggle.com/ebouteillon/top-10-with-first-genetic-algorithm-on-gpu) (version 1 runs 9 hours and scores 0.0450)


# Kissat SAT solver for perfect solution

Kaggle's Conway's Reverse Game of Life 2020 can be viewed as a boolean satisfiability problem. I used the kissat SAT solver, which won first place in the main track of the SAT Competition 2020 and first place on unsatisfiable instances.

The tricky part might be to turn the problem into an efficient CNF form, but it was pretty easy in fact using sympy.

Following notebook solves 9532 puzzles (delta=1) in 25 minutes: [Perfect Solve of Puzzles using a SAT solver](https://www.kaggle.com/ebouteillon/perfect-solve-of-puzzles-using-a-sat-solver)

The drawback of this approach is, it may takes a loooong time for solving some puzzle and bigger the delta, the harder.


# Google OR-tools constraints solver

It is also possible to express the problem as a constraints problem. You will find an iterative approach using OR-tool in section "Notebooks Prize attempt"  below.

The iterative approach consists in solving the puzzle with delta = 1, then solve again this solution with delta=1 until we did it delta times. It is much faster than solving the problem end-to-end, but it does not always find a solution.


# Notebooks Prize attempt

In this competition there is a "Notebooks Prize":

> Highest scoring Kaggle Notebook (as determined by Leaderboard submission score). To be eligible for this prize, the Notebook must do the entirety of its computation within the Notebook.

My solution for this was to use the genetic algorithm running on 1-GPU + 1-CPU (pytorch uses a CPU at 100% even if all the work is on GPU) and an iterative solution using OR-tool on 1-CPU.
The drawback of this approach is: we only have 2 hours and 2 CPU for a GPU notebook versus 9 hours and 4 CPU for CPU notebooks.

2 hours score: : 0.05353 [GA on GPU + OR-tools on CPU2](https://www.kaggle.com/ebouteillon/ga-on-gpu-or-tools-on-cpu2)
9 hours score: : 0.0433 (cannot submit due to code limit)


# Solution ensembling

Ensembling of solutions was pretty easy. We compute the score for each solution found for a puzzle and then we take the solution scoring the lowest.



# Team-up

Teaming with @knstqq , @maximim , @robertjohncannell was a great experience, all keeping the motivation in the team and their solutions to this problem are really amazing. In the end, merging together really is the secret ingredient to get to this position.


# Final notes

Team name comes when the team was only @maximim and I .We had the objective to get under the score .01 what we called "a penny".

Thanks for reading. 😄

My teammates write-ups:
 - [write-up of @knstqq](https://www.kaggle.com/c/conways-reverse-game-of-life-2020/discussion/200539)
 - write-up of @maximim : TBD
 - write-up of @robertjohncannell : TBD
