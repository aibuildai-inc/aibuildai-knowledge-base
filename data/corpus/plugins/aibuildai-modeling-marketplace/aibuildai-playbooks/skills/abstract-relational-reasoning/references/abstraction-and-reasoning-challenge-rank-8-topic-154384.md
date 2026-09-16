# Part of 8th Place Solution - Macro DSL

Competition: abstraction-and-reasoning-challenge
Rank: #8
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154384

Firstly I’d like to thank my teammates for their awesome solutions, help, encouragement and perseverance. This was definitely a competition where combining different approaches helped a lot.

I'd also like to thank Francois and the Kaggle team for hosting this very novel challenge! It's been fun and I look forward to seeing the impact that ARC will have in the future.

Congratulations to @icecuber for a stunning result!

Our team result consisted of three main parts. For my contribution, I used an approach based on a Domain Specific Language. A number of notebooks were shared with good looking DSLs, capable of encoding the solution to a number of tasks, but it quickly became apparent that the program synthesis part was going to be very hard.

It would seem ideal to create a DSL where every command does one small thing well. The power of the approach would then be in the program synthesis. However, given the time constraints for the competition and without a background in program synthesis, I thought I'd try something a little different. By making the commands do more, making them more like macros, the composition of commands became easier. This approach would likely limit the overall generality of the solution, but I hoped that as a proof of concept, it would provide a feasible approach to the challenge within the time constraints.

My initial inspiration came from noticing that many tasks require the input to be split into panels/tiles/objects and then either one panel is selected as the output, or all the panels are combined; for example by a logical operation. The split-filter-combine commands in combination therefore attacked the output smaller than input class of tasks. I then extended the approach to deal with all tasks.

An example program using the DSL, that can be found by the simple search employed is:



For more details, I’ve shared the notebook: https://www.kaggle.com/andypenrose/macro-dsl-for-arc-with-heuristic-search
