# 25th place solution + thought process🥈

Competition: santa-2023
Rank: #25
Source: https://www.kaggle.com/c/santa-2023/discussion/472530

This post outlines our approach which got us 25th

First of all, thanks to @codicon, @timothygao, @alvaroborras and @marksix for teaming up and contributing ideas throughout the competition, and Kaggle for organizing.

[**Notebook: Our final scores by puzzle type**](https://www.kaggle.com/code/yeoyunsianggeremie/santa-2023-25th-place-moves-distribution/notebook)

**Cube**
-----------------------------------------------------

Initially, we utilised public repos, DWalton for edges and RCube for centers. Parity trick was used for AB cubes. These alone got us 23k for 281 and 282, and a 614k 3rd place submission 1 week into the comp. However, it is unable to solve N-type cubes

N-type cubes are more challenging. We first tried to solve it like a normal cube, by recoloring (i.e. for cube 4/4/4, N1 to N16 are labelled as A, N17 to N32 are labelled as B, etc). After solving, we notice that the edges of the cubes are completely solved, only the centers are left. This brings us to the idea of using **cube commutators**.



For example, let’s take those squares underlined in blue. There’s 4 of them in each face. In total 24 in the entire cube. Same goes for the squares underlined in green. **For the centers only, the cube can be reduced into solving many sub-problems of 24 squares.**

We first tried to solve the sub-problems using bidirectional BFS, but that ended up too slow and not feasible, and hence we switched to beam search. Beam Search helped us reduce puzzle 283 from 108k to 27k moves, and also slightly improved the other cubes.

Final solution: Use the DWalton solver to solve the edges (and the centermost square for odd N), then perform beam search on the commutators to solve the centers. 

**Total score for all cubes ~ 132k**


**Globe**
-----------------------------------------------------

Initial Constructive Solution

1. Split the N/M globes into sub-globes of size 2/M. Then solve each 2/M using (2) and (3).
2. Move top pieces in the bottom to the top and vice versa. You can move 1 piece from the bottom to the top and one from the top to the bottom via f0.r0.f0 for example. 
3. Cyclically rotate the top by 1 if the parity of inversions for top and bottom don’t match up. Next, remove inversions via the last commutator in the resource, which swaps a pair in both the top and bottom.
4. Finally, merge solutions by using r0 and -r0 to make all the fx moves f0 moves (fx = x r0’s + f0 + x -r0’s). 
This improved 3/33 (32k -> 13k each). 

To further improve the large globes, we implemented a modified version of the Minkwitz algorithm. The paper for the original Minkwitz algorithm can be found [here](https://core.ac.uk/download/pdf/82526418.pdf), and terms from that paper will be in “quotes”. 

The key differences are as follows:
1. For a vastly faster “Improve” function with comparable results, only choose a single “j”. We chose this j randomly via a weighted distribution where 0..n-1 had weights n..1 respectively.
2. Iterate over “i” in reverse for the “Fill Orbits” function so that words found can be used for subsequent i (this only helped a bit).
3. For the starting word “t”, pick a random word length from 1 to max_start_word_length and then find a random word of that length. The starting words would be too short to be maximally useful otherwise, and repeat words can be useful.

At first, we tried to do a greedy “base” selection, but this led to short term gains at the cost of exponentially increasing “word” lengths for later “table” entries. Instead, the best base was the elements when iterating column by column and bottom to top in each column (we tried a zigzag alternating top to bottom and bottom to top, but this is worse probably due to less order).

Implementation wise, we used C++ for maximum speed. Furthermore, we implemented permutation operations like inverse in place to avoid memory allocation (3x speedup).

Table filling was done until the table appeared to converge as follows. Let the notation for hyperparameters be: (rounds, improvement rounds (“s”), max_start_word_length, new Improve or the paper’s Improve). The reason we switch to the paper’s Improve later is because it tries more combinations of existing short words. This is necessary since it becomes increasingly difficult to find new short words.

3/33:
(1e9, 1e6, 4, fast Improve) (2 hours)
(1e8, 1e6, 32, paper’s Improve) (3 hours)
(1e8, 1e6, 8, paper’s Improve) (2 hours)

8/25:
(4e8, 1e6, 4, fast Improve) (2 hours)
(3e7, 1e6, 32, fast Improve) (20 minutes) (realized it was time to switch to paper’s Improve)
(1e8, 1e6, 32, paper’s Improve) (4 hours)
(1e8, 1e6, 8, paper’s Improve) (2 hours)

All other globes:
(1e9, 1e5, 8, fast Improve) (< 2 hours)

After filling the tables, a solution can be quickly determined via factorization, so we can again use randomization. We applied 1-16 random initial moves for 8/25 and 3/33 and 1-8 for the rest before factorization. We used ~1e7 runs per puzzle, greatly reducing moves (e.g. ~70% for 3/4 and ~20% for 8/25 and 3/33).

This algorithm solves each 3/33 in ~1700 moves, and each 8/25 in ~2500 moves.

**Total score for all globes ~ 26k**

**Wreath**
-----------------------------------------------------

We found the [public hillclimbing notebook](https://www.kaggle.com/code/glazed/humble-hillclimber) from @glazed was extremely useful. It improved almost all of our wreaths, and got the wreath 100/100 down to 2500 moves after many runs, with some code modifications.

**Total score for all wreaths ~ 3.6k**
