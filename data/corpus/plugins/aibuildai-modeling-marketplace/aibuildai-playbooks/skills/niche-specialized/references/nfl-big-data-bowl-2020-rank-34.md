# Public LB 34th Solution Sharing

Competition: nfl-big-data-bowl-2020
Rank: #34
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119475

Thank everyone for the great work. I'm impressed by how elegant the solution of @philippsinger is. Congrats to everyone and I wish our kernels will survive stage 2.

I'll just share some ideas that really helped me.

1. Model: my model structure was mostly based on the kernel shared by @mrkmakr
2. Features: cosine/sine of speed, distance, sum(exp(- defender distance from rusher)) etc
3. Simulate player positions after 1 second and compute derived features
4. Data Augmentation by symmetry (flipping X/Y as most said)
5. Data Augmentation by adding random noise to X/Y, for me uniform random 1 yard helped most
6. CV: GroupKFold by week
