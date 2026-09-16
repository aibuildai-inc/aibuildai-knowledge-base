# 16th Place Solution for Santa 2023 - The Polytope Permutation Puzzle Competition

Competition: santa-2023
Rank: #16
Source: https://www.kaggle.com/c/santa-2023/discussion/472489

# Context

- Business context: https://www.kaggle.com/competitions/santa-2023/overview
- Data context: https://www.kaggle.com/competitions/santa-2023/data

# Overview of the approach

We used bidirectional breadth-first search to optimally solve the small puzzles. The backbone of the larger puzzles were solved with commutators and conjugates, but with several special features (detailed below) that made a big difference.

# Details of the submission

## Solving small puzzles optimally

- We solved the `cube_2/2/2`, `wreath_6/6` and `wreath_7/7` puzzles optimally using simple breadth-first-search (BFS).
- To extend to `wreath_12/12`, we implemented bi-directional BFS (BBFS). As well as enabling us to reach the optimal solution for `wreath_12/12` where we couldn't before (because we ran out of RAM), BBFS was vastly faster than vanilla BFS in finding the solutions and became a standard part of our toolbox for solving parts of other puzzle types. 
- To extend to `wreath_21/21`, we made optimizations to compress the state representation and ran on a box with *lots* of RAM.

## Medium globes

Even the smallest globes weren't solvable by BBFS. What we did here was to solve the "first" N cells using BBFS and then iteratively expand. We did this by masking out the other cells - setting them all to have the same symbol as other (`.`) - which means that many unique states collapse down to a single pseudo-state and the puzzle is therefore much smaller. Once we'd solved the first N cells, we'd extend to N+M cells. We called this Iterated BBFS (IBBFS). It no longer gives optimal solutions, but it gives pretty good ones. We then tweaked it a little further so that N and M were set adaptively - i.e. if BBFS found the solution in very few moves, we'd unwind and make it solve a bigger chunk at once.  If BBFS was taking too long to solve the next M cells, we'd abort and reduce M. We called this Adaptive IBBFS (AIBBFS). AIBBFS was our best-performing method for globes 1/8, 2/6, 3/4, 6/4 and 6/8.

## Large globes

Solving the larger globes proceeded in two phases.  The first phase tried "random" moves to solve as many cells as possible.  The second phase used commutators and conjugates (in Rubik's cube parlance) to solve the remainder without upsetting what was already solved.

### Phase 1

Phase 1 simply used BFS, up to a fixed depth, to find the set of moves that solved the maximum number of cells. Then we did *just the first of* that set of moves and repeated the process. Taking a single step before doing the next iteration of BFS (rather than doing the full set of moves found) improved the efficiency of this phase and markedly improved the total number of cells solvable by this phase.  Given that this phase had an efficiency in the region of 1 move/cell and phase 2 is more like 6 moves/cell, getting more cells solved in this phase was a big win.

### Phase 2

We used BBFS to find a large set of short two-pair commutators - i.e. a small number of moves that swapped cells `{A,B}` and `{C,D}` but left everything else where it was. These commutators was short (8 or 10 moves) but pretty limited in the pairs they could swap. Then we extended them using conjugates - i.e. perform any set of moves as a preamble to bring cells of interest into positions `A, B, C & D` that were covered by a commutator, then use the commutator, then perform the inverse of the preamble.

We built a large (10s of millions of entries) database of pairs that could be swapped in this way. Then, at each step, we searched for the database entry that gave us the best bang-for-buck improvement to the current state (i.e. number of additional cells solved / number of moves taken to solve) and applied that.

(Also worth mentioning something that worked nearly as well and ran much faster. There are some commutators that swap two north-south pairs and other commutators that swap a north-north pair and a south-south pair. For much of the competition we first got all the pieces into the correct hemisphere and then did the within-hemisphere swaps.)

In one of the other write-ups, it sounds like the are also 3-cell commutators available for the globe puzzles. I had wondered about about adding these too, but I was out of time / focusing on helping my teammates solve the largest cubes.

## Cubes

It was primarily my teammates who worked on the cubes, so I have less detail here. But the outline is a lot like the larger globes...

1. Find a prefix that solves as many cells as possible.
2. Use an orbit solver that uses commutators to solve each of the 24-cell orbits independently.
3. For puzzles with wildcards, solve the knapsack problem.
4. Assemble the solutions for each orbit using a travelling salesman solver.

The last couple of phases are different to anything discussed so far.

### Phase 3 - Knapsack solving

Given that phase 2 independently solves each orbit and given that some solutions have wildcards, it may be possible to leave some orbits unsolved. But which orbits should we do that for? This is the traditional [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem). The value of omitting an orbit is the number of moves taken to solve that orbit. The cost (or "weight") of omitting an orbit is the number of cells that will be left in an incorrect state if we don't solve the orbit.

This mostly applied to puzzle #277 which had 176 wildcards (8% of the state). Since it had limited applicability to other puzzles and we had limited time, we just used the straightforward approach of greedily omitting orbits based on their `value/weight` (subject to the "weight" not exceeding the number of wildcards available). There are ways of solving the knapsack problem that get better solutions, but this is often a reasonable starting point.

### Phase 4 - Travelling salesman

Phase 2 solves the orbits independently. Because of that, it's possible that there are moves than cancel out at the end of one orbit solution and the beginning of the next. If we re-order the orbits, we can seek to maximize the total number of moves the cancel out. We noticed that this can be framed as an instance of the (asymmetric) travelling salesman problem. First compute, for each ordered pair of orbits `[A, B]`, the number of moves that can be cancelled ('C') when solving `B` immediately after `A`. The (directed) distance between `A` and `B` is just `-C`. Pass to a [travelling salesman solver](http://akira.ruc.dk/~keld/research/LKH-3/) (which I used for [Santa 2022](https://www.kaggle.com/competitions/santa-2022)) to minimize the distance and therefore maximize the amount of cancellation possible.

We should also have used this method for the larger globes but, again, we were out of time.

# Sources

TODO

# Appendix A: The soft stuff

- **Thanks** to Kaggle and to Ryan in particular for setting this competition and dealing so graciously & efficiently with the little wrinkles.
- Why **Always Day Zero**?  This year I invited a couple of my colleagues to join a team with me. The company that we work for has a culture that it's "always day one" (which you can read more about [here](https://aws.amazon.com/executive-insights/content/how-amazon-defines-and-operationalizes-a-day-1-culture/) if you aren't familiar with the term). I've previously competed in internal capture-the-flag security competitions where obviously "day zero" refers to an exploit that hasn't been patched yet. So, as a little joke, I competed in those competitions under the team name "Always Day Zero" and I've stuck with it ever since.
- I'm really pleased that so many of you benefitted from my **progress updates** and were able to use them to focus your attentions on the puzzles where you were furthest away. Thanks for the encouragements and interesting discussions in the chat. Personally, I find it really adds to the experience.


# Appendix B: Final scores by puzzle type

| cube size | total moves | wreath size | total moves | globe size | total moves |
| --- | --- |
| 2     |     315 (†) | 6     |     150 (†) | 1/8      |    1,104 |
| 3     |    2,821 | 7     |     128 (†) | 1/16     |    1,434 |
| 4     |    4,953 | 12   |     173 (†) | 2/6      |     181 |
| 5     |    5,336 | 21   |     176 (†) | 3/4      |     680 |
| 6     |    3,825 | 33   |     383 | 6/4      |     444 |
| 7     |    2,185 | 100 |     642 |  6/8      |    2,129 |
| 8     |    2,930 | | |  6/10     |   2,673 |
| 9     |    3,810 | | |  3/33     |  11,853 |
| 10  |    4,829 | | |  8/25     |   4,038 |
| 19  |   14,637 | | | | |
| 33  |  41,078 | | | | |

† We know we have an optimal solution.

# Appendix C: Score progression

Here's a graph of our score progression over the competition.



# Appendix D: Final submission

See the attachment for our final submission.
