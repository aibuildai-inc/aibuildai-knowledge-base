# 5th place solution

Competition: arc-prize-2024
Rank: #5
Source: https://www.kaggle.com/c/arc-prize-2024/discussion/550336

Our code is [here](https://www.kaggle.com/code/gromml/arc-prize-2024-poohai-solution).

Basically, our solution consists of 3 ideas:

1) **Ensembling different solutions.**



2) **Applying different postprocessing filters.**



We noticed that an algorithm can make typical mistakes. For instance, a genetic algorithm tends to produce redundant vertical or horizontal lines.







Also, another typical mistake is a wrong shape.



So, you can implement as many postprocessing filters as you can to cover the most common mistakes of algorithms in your ensemble.

3) **Brute-force.** Tasks can be sorted by their identifiers, and the order remains the same across different submissions.



To identify the 26 tasks solved by the [well-known 26 notebook](https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions), you need no more than 100 submissions. Once the 26 tasks are identified, you can try new algorithms one by one (checking whether they are able to solve at least one new task, that is one of the 74 remaining tasks). Once a new task is found, you can identify its ordinal number with the help of binary search. Also, you can leave all other tasks to the strongest algorithm.

**Our summary:**

* A very efficient way to improve an ensemble was to identify tasks solved by the ensemble, and then try to solve other tasks by other algorithms
* It seems that LLMs could be added to our ensemble, but fine-tuning is needed
* Selecting the right attempt (the correct answer) is difficult, while removing wrong attempts is easier
* An algorithm can make typical mistakes (like the genetic algorithm often produces redundant lines, and you can create the corresponding postprocessing filters)

Congratulations to all the winners, and thanks to the host team for organizing the competition! Looking forward to ARC Prize 2025!
