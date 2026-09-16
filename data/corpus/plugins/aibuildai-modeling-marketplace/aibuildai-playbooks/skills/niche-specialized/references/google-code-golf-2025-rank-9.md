# 9th place write-up

Competition: google-code-golf-2025
Rank: #9
Source: https://www.kaggle.com/c/google-code-golf-2025/writeups/9th-place-write-up

## Acknowledgments

We sincerely thank the organizers for hosting this remarkable competition, the participating teams for sharing their knowledge, and the broader code-golf community for their open contributions of techniques and insights.
We are especially grateful to lynn, @kayjoking  and @garrymoss for their outstanding contributions, which greatly inspired and informed our work throughout the contest.

## Overview

Our team consisted of five members who met through competitive programming. We manually implemented and optimized every single task in this competition.
Although none of us had prior experience with code golf, we devoted over 1,000 hours across three months, learning continuously from AI tools and the broader code-golf community as we refined our approaches and deepened our understanding.

## Solutions / Reproducibility
Our final submission is available in the following GitHub repository:

https://github.com/4eta/google-code-golf-2025

Each individual solution can be found under the src/task directory.


## Approach

### Overall Approach
Most of our code was written and refined manually. Unlike the top-ranking teams, we had very little prior experience with code golf. Throughout the three-month competition, we continually improved our solutions by learning from AI tools such as ChatGPT, Gemini, and Claude, as well as from the wealth of knowledge shared by the existing code-golf community.!

Our timeline and focus evolved as follows:
- August: Focused on solving all 400 tasks without prioritizing code length.
- September: Studied short public-best solutions to learn core code-golf techniques.
- October: Applied everything we had learned, working tirelessly to reduce every solution by even a single byte.

We believe our collective dedication—over 1,000 hours across three months—was the key to our gold medal performance.


### Manual Optimization Process

We managed our progress using a custom dashboard (see below), which tracked all problems and submissions.


This dashboard allowed us to prioritize tasks based on several perspectives, such as:
- Problems where the public-best solution was significantly shorter than ours.
- Tasks with a large score gap between our submission and the public-best.
- Problems that were recently or rarely updated within the team.
- Cases where we were outperformed by lower-ranked teams on specific tasks.

By sorting problems according to their latest update times, we encouraged a process we called “chasing” — improving upon each other’s partially completed ideas. This dynamic cycle of refinement not only led to better code but also fostered a strong sense of teamwork and momentum throughout the competition.

### Techniques

we learned extensively from the broader code-golf community. Many creative techniques—such as rotation and recursion tricks like the example below—were invaluable in shaping our approach:

```py
p=lambda m,k=3:-k*m or p([[y for y in x]for x in zip(*m[::-1])],k-1)
```

For gathering knowledge and inspiration, we found the following resources particularly helpful:

* [https://www.codingame.com/blog/code-golf-python/](https://www.codingame.com/blog/code-golf-python/)
* [https://code.golf/wiki/langs/python](https://code.golf/wiki/langs/python)
* [https://codegolf.stackexchange.com/questions/54/tips-for-golfing-in-python](https://codegolf.stackexchange.com/questions/54/tips-for-golfing-in-python)
* [https://github.com/kajott/adventofcode](https://github.com/kajott/adventofcode)

We also relied heavily on the following repositories and tools while writing and refining our code—We are deeply grateful to their creators.

* [https://github.com/google/ARC-GEN/tree/main/tasks/training](https://github.com/google/ARC-GEN/tree/main/tasks/training)
* [https://github.com/lynn/pysearch](https://github.com/lynn/pysearch)
* [https://www.kaggle.com/code/garrymoss/compressed-variable-name-optimization](https://www.kaggle.com/code/garrymoss/compressed-variable-name-optimization)
* [https://lynn.github.io/flateview/](https://lynn.github.io/flateview/)



## Conclusion
Fun to solve, painful to rank.
