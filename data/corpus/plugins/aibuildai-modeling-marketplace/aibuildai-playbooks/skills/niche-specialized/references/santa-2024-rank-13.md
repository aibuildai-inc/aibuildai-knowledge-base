# 13th Place Writeup

Competition: santa-2024
Rank: #13
Source: https://www.kaggle.com/c/santa-2024/discussion/560683

First of all, thank you to the organizers for creating such interesting problems every year. This year, in particular, the same score was exactly the border for the gold medal. I think it was a problem of just the right level of difficulty. (We may not have won the gold medal, though :D)
And thank you to everyone who participated in the competition. The public discussion and code were very insightful, and were fair.
And thank you to all my teammates. We are a team from the University of Electro-Communications in Japan. hoxosh( @cashfeg ) and yukari17( @yukari17 ) are faculty members, Prgckwb( @prgckwb ) is a master course student, and kumanomi( @kkkggg ) is an alumnus of the extension program on data science.

Santa Master is the name I often use, but it was taken this year, so I decided to call myself 本家Santa Master  (本家 means the original family within extended family in Japanese).

## Overview

We, especially me (hoxosh), tried out a lot of things, and we together improved on the ones that worked. Prgckwb managed GPU resources and we checked progress on WandB. In the end, kumanomi’s one for Simulated Annealing (SA), prgckwb’s one for Genetic Algorithm (GA) and yukari17’s variant of Local Search with kick (Sano Weapon). were good ones. 
It took a long time, but we were able to get the same scores as the published top scores on ids 0, 1, 2, 4, but we got stuck on id 3 and 5. 
For id3, after many trials to get out of the local optimum, we came up with the idea of fixing the first word, and we got 195 using SA with the first word fixed as magi, and we were able to get 191.73 using GA (1/23).
As for id5, we had been stuck at 32 for a long time (until 1/29). Even after reading the [code](https://www.kaggle.com/code/woosungyoon/alphabetical-sample-5) that had been accidentally released, we could not immediately understand it. It was only after we tried that code for a day and got 31.5 that we finally understood it (49.5 hours before the deadline). From there, Sano Weapon quickly pushed up the score, and then SA and GA converged to reach 246.82532 in several hours. After being stuck again for half a day there, and being overtaken by one team, SA made the final touch (25.5 hours before the deadline).

## Simulated Annealing
(kumanomi part)
- Simulated Annealing Foundation: Developed a method to reduce the likelihood of getting stuck in a local optimum, specifically targeting a local optimum of 197.5 for the ID3 problem.

- Initial Setup:
    - Base Parameters:
        - Start Temperature: 10
        - End Temperature: 0.5, 0.1
        - Cooling Type: Exponential
        - Total Steps: 100,000
    - Neighborhood Operations:
        - Common: word_insert, words_swap, words_swap_3, etc.
        - TSP-inspired: 2-opt, 3-opt, Or-opt, etc.

- Performance and Strategy(ID3):
    - Achieved an initial score of 197.5, but reaching 195 took significant time.
    - Implemented a cutoff strategy: If the score didn't reach 220 by 60,000 steps, the process was restarted.
    - Emphasized rapid annealing and a high number of trials to improve results.
    - Each trial lasted approximately 30 minutes on the preferred machine.

- Outcome: Successfully achieved a score of 195 and passed the task to @Prgckwb for further development.

## Genetic Algorithm
( @Prgckwb part)
I contributed to late-stage improvements by using a genetic algorithm, achieving `195.0→191.7` at id=3 and `28.8→28.5` at id=5. The algorithm was guided by the following principles:

- Use multiple diverse existing local solutions as initial individuals
- Create offspring text by crossover from two parents, as shown in the diagram below
- Perform mutation by rearranging text similarly to Kumanomi ( @kkkggg ) 's SA
- Switch to SA if no improvement is made after a set number of iterations


## Iterated Local Search with kick. 
( @yukari17 part)
Mainly helped to get the score from 31.5 to 28.8(id=5).
1. Local search
  - Divide the sentence into 3 parts and perform reordering(all patterns): O(N^2)
  - Swap Words
  - two next to each other(all patterns): O(N)
  - swap with n spacing: O(N)
  - word insertion(all patterns): O(N^2)  
Search that allows up to a worsening of the score of the difference X, not the transition probability(acceptance probability). 
X is gradually lowered(it thoroughly on low heat). 
Aim is to escape from local solutions
2. Sano Weapon Kick(”Sano” is the name of yukari17)
Weapon not to break too much. But break it to some extent
In fact, it was executed head part and tail part

3.Same score is not passed twice
Scores that have passed in the past are excluded from candidate neighborhoods
