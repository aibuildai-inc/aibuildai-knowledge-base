# 1st Place Solution

Competition: santa-2023
Rank: #1
Source: https://www.kaggle.com/c/santa-2023/discussion/472405

Solving 10x10x10 cube (272) in 454 steps[Solving 10x10x10 cube (272) in 454 steps]

Among the three types of puzzles, the wreath puzzle was not particularly focused on because a very short solution could be found using simple beam search, and it had a lower total score compared to the others, making it a lower priority. Therefore, we concentrate on the solutions for the remaining two types, the cube and the globe puzzles. The strategies for solving both puzzles are based on the same principle.

Due to each move affecting a large number of pieces at once, simply applying moves as is can lead to a somewhat organized state, but fully solving the puzzle becomes extremely challenging. Thus, the key is to find sequences of moves that only swap a few pieces while leaving the rest unchanged. Remarkably, for both puzzles, there exist sequences of moves that only rotate three pieces and leave the rest unchanged, referred to as 3-rot.

- Example for the cube puzzle: d3.f2.d2.-f2.-d3.f2.-d2.-f2
- Example for the globe puzzle: f0.r0.f0.r1.f0.-r1.f0.-r0

When considering the cluster decomposition of all pieces (a cluster being a set of pieces that can be interchanged through moves), it turns out that for most clusters, except for special parts like the cube's corners, there exists a 3-rot for any three elements within them.

- For the cube puzzle: Corners, centers of faces, and centers of edges are exceptions. For other clusters, taking symmetry into account, it suffices to consider the diagonal parts of a 4x4x4 (1,1), the cross parts of a 5x5x5 (1,2), the non-diagonal and non-cross parts of a 6x6x6 (1,2), and the edge parts of a 4x4x4 (0,1). For each, a bidirectional search was performed to enumerate all shortest 3-rot for any three elements, with the longest being 14 moves.
- For the globe puzzle: The central row when the number of rows is odd is an exception. For other clusters, with the number of rows as 2 and the number of columns as 2c, R=f0.r0.f0.r1.f0.-r1.f0.-r0 corresponds to a 3-rot of ((0,0) (1,c) (1,c-1)). A 3-rot for any three pieces can be found by first finding a sequence of moves A that moves those three pieces to a state where R can be applied (where other pieces can move freely), using breadth-first search, and then the 3-rot can be obtained by A.R.-A.

When solving each cluster using 3-rot, it is important to note that all 3-rot are even permutations, so the overall permutation must also be even. Combining these insights, we can use the following approach:

1. Solve the special parts.
2. Operate without disrupting the solved parts so that all remaining clusters become even permutations.
3. Solve each cluster independently using 3-rot.

To achieve shorter solutions, we employ the following key ideas:

- Since 3-rot requires at least 8 moves and is lengthy, it's more efficient to bring the puzzle to a somewhat solved state using elementary moves or short sequences before employing 3-rot for the final touches.
- When adding a new sequence B to an existing sequence A, canceling out the end of A with the beginning of B can shorten the overall sequence. For example, if A=A'.ri and B=-ri.B', then A.B becomes A'.ri.-ri.B'=A'.B', thus saving 2 moves.
- If the current sequence is A=a[0]...a[T-1], instead of appending a 3-rot B for (i j k) at the end to form a[0]...a[T-1].B, at arbitrary time t, inserting a 3-rot B' for some (i' j' k') to achieve a[0]...a[t-1].B'.a[t]...a[T-1] results in the same state. Selecting the appropriate time t can make B' shorter than B or result in more cancellations. Therefore, rather than constructing the sequence from the front, it is better to try the insertions for all times and select the best time.

Based on these ideas, we use the following approach:

1. Solve the special parts.
2. Use elementary moves to make all clusters even while roughly aligning them.
3. Bring to a somewhat solved state using short sequences.
4. Insert 3-rots at arbitrary times.

The details of each part and the score table can be found in our repository: https://github.com/wata-orz/santa2023_permutation_puzzle
