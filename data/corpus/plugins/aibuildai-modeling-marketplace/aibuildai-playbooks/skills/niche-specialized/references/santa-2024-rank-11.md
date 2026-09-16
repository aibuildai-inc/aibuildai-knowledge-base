# 11th place solution

Competition: santa-2024
Rank: #11
Source: https://www.kaggle.com/c/santa-2024/discussion/560542

# Introduction
Thank you to the organizers for hosting such an exciting competition.  I would also like to express my gratitude to the other participants. In the final stages of the competition, competing together for a gold medal was very thrilling and an enjoyable experience. I also learned so much from the Notebooks and Discussions, which were a tremendous help in achieving this result. Thank you very much.

# Summary
I began my improvements from a submission file with a score of 250.10105 that was publicly shared in [this notebook](https://www.kaggle.com/code/veniaminnelin/fine-tuning-word4-simple-permutation?scriptVersionId=218441008). As a result, ID=0,1,2 had already achieved optimal solutions on the leaderboard, so I only needed to work on ID=3,4,5. This allowed me to conserve computation resources for exploration. Thank you to @veniaminnelin for sharing this notebook. I would also like to express my gratitude to the author of the notebook referenced by this notebook.

All of my solutions relied entirely on the simulated annealing, and how to generate neighboring solutions was a crucial factor. I will describe the details below.

# Generating Neighboring Solutions
Since using "swap" relatively disrupts the original solution, choosing "insert" as the method for generating neighboring solutions was the starting point for improvements.
The methods I used are as follows.

### 1. random insert
- Randomly select i and j (0 ≤ i, j < len(words)), and then insert the word at position i into position j.
- Used in ID-3,5.

### 2. distance-based probabilistic insert
- Randomly select i (0 ≤ i < len(words)). Then, based on the distance from i, select j probabilistically as the insertion position, and insert the word at position i into position j.
$$
d_j = |i - j|, \quad p(j) = \frac{\frac{1}{d_j}}{\sum_j \left(\frac{1}{d_j}\right)}
$$
- Used in ID-4,5.

### 3. variable-length random insert
- A method that modifies random insert to a variable-length approach.
- The value of length was determined according to the following probability distribution
    - p(L=1) = 0.4, p(L=2) = 0.3, p(L=3) = 0.2, p(L=4) = 0.1
- Used in ID-5.

### 4. alphabetical insert
- Randomly select i (0 ≤ i < len(words)). Then, from the group of words whose first letter is the same as that of word i, randomly select one and insert word i either before or after the selected word.
- Used in ID-5.

### 5. partial permutation
- Randomly select i (0 ≤ i < len(words) - L + 1), and take a permutation of the length L(=4) sequence of words starting at i as the neighboring solution.
- Used in ID-5.

# Details

### ID-3
- For ID-3, instead of using the solution from the public notebook as the initial solution, I started with a sequence initialized in random order.
- By using the random insert method described above, I reached a score of 197.5 after several runs.
- However, as noted in this discussion, this is known to be a local optimum and is quite difficult to escape.
- To break out of it, I tried fixing the first word and applying random insert. By placing “magi” as the first word, I managed to escape the local optimum and achieve 191.7.
- Parameters
    - n_iterations per run : 100,000
    - Cooling Schedule: Linear Decrease
    - start_temp : 10
    - end_temp : 0.1
    - batch_size : 8, 16, 32

### ID-4
- Among these three IDs, ID-4 was relatively easy. By using the solution shared from the public notebook as the initial solution and applying distance-based probabilistic insert, I was able to reach a score of 67.54 in a single run.
- Parameters
    - n_iterations per run : 100,000
    - Cooling Schedule: Linear Decrease
    - start_temp : 2
    - end_temp : 0.1
    - batch_size : 32

### ID-5
- As with ID-4, I used the solution from the public notebook as the initial solution and tried distance-based probabilistic insert, which improved the score to 31.66.
- Next, using the 31.66 solution as the initial solution, I ran several searches combining random insert, distance-based probabilistic insert, alphabetical insert, and partial permutation, ultimately improving the score to 30.70.
- Upon examining the solution at this point, I observed that there were three alphabetically ordered blocks. Therefore, I modified the initial solution—derived from the best solution at that time—so that the number of these blocks ranged from one to four, and then performed the same search as before. As a result, having two blocks yielded the best performance, improving the score to 28.57.
- Finally, I changed random insert to variable-length random insert. In the 28.57 solution, I noticed some sequences that didn’t follow alphabetical order (e.g., “card game” and “wrapping paper”). I thought that if there were phrases of three or more words forming a single semantic unit in the optimal solution, random insert alone wouldn't be able to handle them. By making this change, I was able to reach a score of 28.52. (In fact, it appears that the “and and and” in the stopwords section needed to be placed between blocks.)
- Parameters
    - n_iterations per run : 50,000~100,000
    - Cooling Schedule: Linear Decrease
    - start_temp : 0.1~1
    - end_temp : 0.01~0.1
    - batch_size : 16, 24

# computation resource
- I rented A100 in the cloud for 10 days before reaching the LB-optimal solution.
- For the first five days, I used a single A100 for the remaining five days, once improvements started, I used A100×3.
