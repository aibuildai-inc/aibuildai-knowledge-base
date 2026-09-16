# [23rd solution] TanakaAI`s part.

Competition: santa-2024
Rank: #23
Source: https://www.kaggle.com/c/santa-2024/discussion/560616

First of all, I would like to thank all my team members @felixmneumann , @asimandia , @veniaminnelin ,  and @ivanisaev. I would also like to thank all the amazing competitors from whom I learned a lot. By the way, this is my first medal and my first experience working with a team.

General Approach
Like many others, I used a large run to generate good candidate solutions, and then conducted smaller local runs to optimize these candidates. The large runs were mostly performed using simulated annealing (SA) with a function to fix certain indices (positions). There were two small runs: one employed a local window search (thanks to @jazivxt's notebook), and the other involved replacing n-grams throughout the sentence (for n up to 4).

Specific Approach
For ID 4, I computed and stored strings and perplexity (ppx) in a CSV file, for saving recomputation cost (though this was not the fastest method), which resulted in a 3 GB file. Once the perplexity dropped below 70, I fixed the first 10 and last 4 words and let the SA run
