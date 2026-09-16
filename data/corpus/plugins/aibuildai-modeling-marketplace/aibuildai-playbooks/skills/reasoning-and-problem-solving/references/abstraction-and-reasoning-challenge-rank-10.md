# Our 10th place solution [with code]

Competition: abstraction-and-reasoning-challenge
Rank: #10
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154294

Congratulations to everyone that the competition is finally over! Just after the end I would like to tell you a bit about the solution of our team “Puzzlemaster” that got 10th place.

Our solution is an ensemble of 3 parts that solved 3 sets of tasks that has a partial intersection:
- DSL-algorithm implemented by @felipefonte99 
- ML-algorithm implemented by @nikitaovsov. 
I think they will create other posts about their solutions or post comments in this thread.

- Genetic algorithm that tries to fit a DSL-program that was created by me and @artyomp. It solved 10-12 tasks. 
Now I am going to tell you about it.

At first I would like to thank authors of these kernels that inspired me from the algorithm’s perspective:

[https://www.kaggle.com/arsenynerinovsky/cellular-automata-as-a-language-for-reasoning](https://www.kaggle.com/arsenynerinovsky/cellular-automata-as-a-language-for-reasoning)
[https://www.kaggle.com/zenol42/dsl-and-genetic-algorithm-applied-to-arc](https://www.kaggle.com/zenol42/dsl-and-genetic-algorithm-applied-to-arc)

And of course the kernel that I opened 1000 times:

[https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines](https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines)

**The solution**

We created a Domain-Specific Language based on commands that can be applied to a grid.
There are many commands in our language. Each command has a couple of parameters that are randomly sampled during training. We call it “global rules” (in opposite to cellular automata rules that are described below)
Examples of the commands:
- Draw a line (or some color) from cells (of some color) (until it stops at some color).
- Map all cells of (some color) to (some color)
- Rotate the grid
- Apply gravity to (figures/cells) of (some color) in the direction of ((some border)/some color)
- others
All that written in brackets are randomly sampled parameters.

At the end of the competition we added elements of “Multiple grids” to the DSL. For example, it could split the grid (for example, by figures on it), apply some commands to some of the grids and then merge them back. This boosted our LB score a lot.

There were also “Cellular automata” commands that were applied after previous commands. More details about cellular automatas can be find at this kernel:
https://www.kaggle.com/arsenynerinovsky/cellular-automata-as-a-language-for-reasoning

Our DSL contained a couple of randomly parametrized CA-rules. For example, checking of the direct neighbors, checking of colored corners and so on.

So the final DSL-program is a sequence of “global” rules, then CA-rules are applied to the result of that.

**The C++ part**
Very important part of our solution was that the “global” rules and “CA-rules” were implemented by @artyomp using C++. He did a great job that accelerated the whole algorithm at 100 times and allowed us to create many new rules and not to care much about the complexity of the search space. Without C++, we would probably solve half of the test tasks we solved.


**The search algorithm**
In fact, it’s similar to the algorithm from https://www.kaggle.com/zenol42/dsl-and-genetic-algorithm-applied-to-arc but with a number of heuristics.
Our algorithm creates random sequences of programs, creates asexual mutations (tries to replace commands with new commands, insert new commands, delete commands) and "sexual" mutations (tries to concatenate random prefixes and suffixes of 2 programs with some mutation in the end).

**Quality measurement**
How to measure if the program is good? Obviously, we should measure accuracy (how many cells are correct). If it equals to 1 for all training pairs, we found a solution. But since we use genetic algorithm with mutations of 1 rule to find a solution, we might need more metrics to allow the population to be bigger. My observation was that we should be careful with that because if we add more metrics, it is more likely that a junky program with many commands will be created by the algorithm that can solve training samples but overfits. 

So as a result, our final choice of metrics:
- Mean accuracy for all training outputs
- Accuracies for each output
- The mean of the following binary metrics: for each color from 0 to 10 we compute if our prediction of the color is absolutely correct (in terms of both precision and recall). It helped the algorithm to find correct sequences that fit correctly colors one by one.

**The code**
The DSL-solution is about 2.5k lines of code. We are going to clean it a little bit, make some comments and publish as a kernel so you can dive into details.
I didn’t mention a huge number of details of the algorithm because there would be too much text. But I am ready to answer your questions about the algorithms if you are curious.

UPD.

The C++ version is in repo:
https://github.com/artyompal/kaggle-abstract-reasoning

The kernel with the python version:
https://www.kaggle.com/alexfritz/genetic-dsl-part-from-10th-place-python-version

**Conclusion**
I would like to thank organizers and my team @artyomp @nikitaovsov @felipefonte99! It was a great experience for me and I learned a lot of new things.
