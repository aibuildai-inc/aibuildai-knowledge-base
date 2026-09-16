# 12th place solution

Competition: santa-2023
Rank: #12
Source: https://www.kaggle.com/c/santa-2023/discussion/473094

result: [95484.csv](https://gist.github.com/sash2104/fc2628f2ce6770e92c827e799a9726d1#file-95484-csv)

First of all, thanks for organizing the competition. This is my third consecutive year participating, and I've enjoyed this problem the most.
I'll mainly discuss the large-size Cube problem, as it took up most of our time.

## Overview of the Approach
For all wreaths and small-size cubes, solutions were obtained either through bidirectional BFS or simple beam search.

For all globes and large-size cubes, the following steps were taken:
1. Solve the special parts. 
  - The main objective is to resolve parity.
2. Beam search with moves that preserve parity as candidate moves.
  - The foundational strategy involved using 3-rot commutators as candidate moves, and for further improvement, the pool of candidate moves was expanded.

The solutions obtained were further optimized using public notebooks such as [here](https://www.kaggle.com/code/shitovvladimir/optimize-any-solution-with-group-theory-approach) and [here](https://www.kaggle.com/code/glazed/humble-hillclimber), and when applicable, the best solutions from public notebooks were adopted.
We are grateful for the valuable insights gained from utilizing these public notebooks.

## Scores
```
total   95484
cube    76719
wreath  1472
globe   17293
cube_2/2/2      315
cube_3/3/3      2961
cube_4/4/4      6314
cube_5/5/5      6172
cube_6/6/6      3528
cube_7/7/7      2007
cube_8/8/8      2738
cube_9/9/9      3556
cube_10/10/10   4277
cube_19/19/19   12021
cube_33/33/33   32830
wreath_6/6      150
wreath_7/7      128
wreath_12/12    173
wreath_21/21    201
wreath_33/33    210
wreath_100/100  610
globe_1/8       961
globe_1/16      935
globe_2/6       205
globe_3/4       954
globe_6/4       463
globe_6/8       1351
globe_6/10      1392
globe_3/33      8288
globe_8/25      2744
```

## Terminology
- facelet: The smallest unit of state. (e.g. There are 6534 facelets in cube_33/33/33)
- cubie: An individual small cube in a Cube puzzle. (e.g. In cube_33/33/33, there are 5766 center cubies, 372 edge cubies, and 8 corner cubies)

## Large-size Cube Solution
1. Solve the special parts.
   - Use the moves that appear an odd number of times in the sample_submission.csv to resolve parity.
     - Since executing the same move an even number of times does not change parity, arranging only moves that appear an odd number of times can always resolve parity.
     - The sequence of moves obtained is manually fine-tuned to reduce the number of moves further (e.g., `r1.d1` is removed because it does not change parity).
   - For cubes with an odd N, also align the positions of the central six cubies.

   Examples: For id=283, use `f0.f16.f2.f4.f5.f6.f11.f12.f13.f14`; for id=257, use `f3.r3.f1.f2`.

2. Beam search with moves that preserve parity as candidate moves.
   - The beam width varies with N, ranging from 1 to 100.
   - The evaluation metric is the fewness of moves. The number of matches with the solution state on a cubie basis is considered equivalent to turns.

   The candidate moves used are as follows:
   - Moves that do not disturb parity in two steps.
     - Example for N=33: `r1.-d31`
   - All combinations of 3-rot.
     - After enumerating all 8-move commutators of 3-rot (e.g. `R=-d1.-r0.d0.r0.d1.-r0.-d0.r0`), use breadth-first search to find the remaining moves in the form of `A.R.-A`.
   - Commutators of 4 to 8 moves that alter a small number of facelets.
     - Examples for N=33: `f1.r1.-f1.-r1`, `-d32.-r31.d22.r31.d32.-d22`
     - Adjusted based on N. For N=5, all 4, 6, and 8 moves that alter up to 24 facelets; for N=33, all 4 and 6 moves that alter up to 12 facelets.
   - Commutators of rotations of two corners and rotations of two central edges.
     - Example of corner rotation moves for N=33: `r0.f0.-r0.-f0.r0.f0.-r0.-f0.r32.f0.r0.-f0.-r0.f0.r0.-f0.-r0.-r32`
     - Example of edge rotation moves for N=33: `-r16.d0.r16.-d0.-r16.d0.d0.r16.-d32.-r16.-d0.-d0.r16.d0.-r16.-d0.r16.d32`
   - Commutators composed of the above.
     - Compositing multiple commutators can reduce the number of moves.
     - Example: `-r1.r2.d0.f1.-f2.-d0.r1.-r2.d0.-f1.f2.-d0` is a composition of four 3-rot such as `-r1.d0.f1.-d0.r1.d0.-f1.-d0` and `r2.d0.-f2.-d0.-r2.d0.f2.-d0`.

## Globe Solution
1. Solve the special parts.
   - Roughly align positions using only `r{i}` moves.
   - Most cases were solved without explicitly resolving parity; for unsolvable cases, a few random moves were added before reattempting.

2. Beam search with moves that preserve parity as candidate moves.
   - The beam width varies with N, ranging from 1 to 10.
   - The evaluation metric is the fewness of moves, with the number of matches with the solution state on a facelet basis considered equivalent to turns.

   The candidate moves used are as follows:
   - All combinations of 3-rot. (e.g. `r0.f0.r1.f0.-r1.-f0.-r0.-f0`) 
   - Commutators of 4 to 12 moves that alter a small number of facelets. (e.g. `r1.f3.f4.-f3.-r1.-f4`)
