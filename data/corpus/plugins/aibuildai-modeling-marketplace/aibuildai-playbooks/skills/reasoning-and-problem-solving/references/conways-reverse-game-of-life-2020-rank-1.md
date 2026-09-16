# 1st place write-up, Deeper Systems

Competition: conways-reverse-game-of-life-2020
Rank: #1
Source: https://www.kaggle.com/c/conways-reverse-game-of-life-2020/discussion/200980

# Deeper Systems Write-Up

First I want to congratulate everyone for participating and particularly to "Under a penny" for the intense competition they brought about at the end :) 

A lot of factors led to our ultimate win. It was a combination of good process, teamwork, and some improvisation at the end.

## The Process

Our approach was essentially to try as many things as possible and put our processing power on whatever the most promising one was at the time. 

A key component was a small web service we built where our scripts could request a random individual to try, and then report their success or failure along with the algorithm name and the time it took to process. We then had a small webpage listing the algorithms and their performance in terms of score improvement per hour. This had several advantages:

- we could track how each algorithm did
- we could easily plug in new ones without needing to re-code timing and performance stats on each separate attempt
- it let us distribute the latest changes quickly to whatever available cores we had

It's also helping me write this now as it helps to remember all we tried :). There are 155 rows on the page as I look at it now, and we tried a few more things before we got the service running.

## Approaches

To put it in technical terms, we tried "a whole bunch of stuff". What we found is that certain approaches would work really well at the start and then get diminishing returns as they solved the 'easier' versions of the problems, or reached the limit of the approach. Thus we were constantly adding new approaches and rebalancing our power based on their continued performance. These are listed roughly in chronological order.

### Simulated Annealing

This is your standard simulated annealing, implemented in C++. It ran super-quick and got large improvements at the start. We did it such that each 'run' would try a few thousand epochs on one sample, then report the improvement (if any) and get a new sample to try on.

### Genetic Algorithm

We ended up doing three separate implementations of a genetic algorithm. The first one in C++ didn't get many improvements. We were going to abandon the approach but then took heart in seeing [Desareca's success](https://www.kaggle.com/desareca/game-of-life-genetic-algorithm-spanish) and implemented another one in Python. This got rapid improvements early on but started to slow down quickly. As a last optimization we translated the code directly from Python to C++ and got a ~50x pickup on the approach. We kept this one running for a while.

### Template Matching

We tried generating all possible 3x3, 4x4, and 5x5 configurations, running them 1, 2, 3, 4, 5 steps, caching the results, and then 'pasting' these in to exact partial matches. It was a bit finicky and didn't work out that well. Our hope was we could paste multiple of these and get a good partial score, but it proved the problem progressed too chaotically to be able to segment it like this. We abandoned it for other approaches.

### Z3 Solver

We then learned about the SAT solvers, thanks to James McGuigan's [excellent notebook](https://www.kaggle.com/jamesmcguigan/game-of-life-z3-constraint-satisfaction). This was a new tool for us. Here we experimented with trying to find what gives the most improvement: trying the 'easiest' problems first, or the ones that we had the 'most error' on from our other approaches. We quickly discovered that the fewer pixels a stop board had, the easier it was to solve for the solvers, so we prioritized running those first. 

### Picosat Solver

Some time after the Z3 solver we discovered Picosat, thanks to [Serhii Hrynko's notebook](https://www.kaggle.com/elvenmonk/pycosat-exploration). Contrary to what I read on some other reports, we found Picosat to be much faster than Z3. 

We tried both approaches of doing an all-in-one solution (i.e. given stop board find the
board 3 steps back), or trying the iterative, get one step back, then another, etc. We found it was much, much more effective to do the iterative approach. The key was to add additional constraints in order to minimize the number of pixels the solver found on a board. Fewer pixels means more likely to get a father that has its  own father, and quicker for the solver to solve. We implemented a search algorithm that would use binary search to find the **number** of dead cell constraints to add, and then it would get a random subset of these and yield solutions. If the solutions failed it'd loosen the constraint (requiring fewer dead pixels) and if the search failed with no constraints we
backtracked one level up.

Very effective was imposing a time-out on the solver. At first we gave them a 1 minute timeout and solved all we could within a minute. Then we expanded to 10 minutes, then an hour, and finally by the end we were running them with a 2-hour timeout before aborting. The effect of the timeout was to essentially re-start the search a different first-step-back board. From  observation we found the solver would get stuck down a hopeless branch, either one that has no eventual solution, or one where it might but it's so computationally intensive to try each step that it would take forever to run.

### Partial Solvers

The more the solvers ran, the greater diminishing returns we found. We experimented a lot and eventually found something that would help, namely to zero-out the edges of the stop board and have the solvers partially solve this. This would generate a start board that gets the center of the solution fully correct, and then our iterative approaches would pick them up and gradually improve them. 

The inspiration from this came from seeing that as the boards evolve, the solvers are very quick up until the wrapped-around edges 'touch' each other, then it becomes very difficult to find a solution. The hardest cases at the end were the ones that looked sort of like a cosmic background radiation where everything was very dense and uniform. It's interesting that there were 5 delta-2 problems and 129 delta-3 problems that eluded solving until the very end.

### Incremental Exhaustive Search

At this point we joined forces with the formidable Vlad Golubev, which gave us a huge boost to our score right away. I will link to his write-up once it's available, but to summarize here he implemented a crazy-fast C++ exhaustive search algorithm. The approach was to flip each pixel, pick the best board and then continue until there's a board with no single-pixel  improvement. Then he would flip a random 5 pixels and restart from there. The huge speed-up he found was by, whenever he flipped a pixel, only re-calculating the affected neighbors. 

Thanks to our web service set-up it was very easy to plug his algorithm in and keep it chugging alongside all the other ones we tried.

### Kissat

As the competition got stiff at the very end we scrambled to find another magical pick-up. If picosat was so much faster than Z3, maybe another SAT solver would be even faster? 

We tried cryptosat, microsat, and kissat. Interestingly there was no one solver that performed the best on all problems.  Cryptosat was generally slower although on some small sets would give a ~10% pick-up. Microsat was faster than  picosat on small instances, but slowed down on harder problems. Kissat was much slower on small instances than picosat.  We were about to give up but then we saw Kissat solve a problem in 5 seconds that took picosat 60 seconds to solve. There was hope yet!

We plugged kissat instead of picosat and let it run and we started getting huge gains again. By this point we had only the harder of the problems left to solve so kissat was a clear gain on all of these. Here I want to personally thank the SAT solver community for having everyone take input in DIMACS form, it made it easy to plug-and-play different solvers :) . 

### Final Optimizations

As the clock ticked down we tried changing the order of the problems again. We managed to solve the remaining delta-2s, then distributed some cores on delta-3, some on delta-4, and others to pick random from what was left. 

## Conclusion

As future work, it would be interesting if we could get to 0. The largest optimization I could think is to improve the search on the solvers, particularly when to backtrack. As the larger problems take so long to solve it made it hard to tweak this, especially near the end where every restart could lose us up to 2 hours of progress on each problem. The other thought was to add constraints to "de-cluster" solutions, e.g. to not have a thick cluster of 2x3 pixels or a 5-pixel-long row or line. We experimented with this but didn't have time to get anything conclusive at the end.

At the end of the day we made it through. I'd like to thank my colleagues at Deeper Systems and a special thanks to Vlad for teaming up with us and helping us get to victory :).
