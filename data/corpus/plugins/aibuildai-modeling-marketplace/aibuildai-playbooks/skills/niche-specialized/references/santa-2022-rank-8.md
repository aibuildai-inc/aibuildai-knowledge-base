# 8th place solution

Competition: santa-2022
Rank: #8
Source: https://www.kaggle.com/c/santa-2022/discussion/379511

We, @qiaoshiji, @vialactea, @zyu2017 and @runningz, thank the Kaggle team for this intriguing problem and all fellow competitors! Our solution is in a rather brutal force fashion and we briefly summarize our intuitions in this thread.

As many other teams do, we first find a good initial TSP tour and then make it configuration friendly. Our solution can be summarized roughly into four stages
1. Finding a good initial TSP tour
2. Manually freeze a few edges
3. Optimizing TSP with our choice of edges
4. Convert TSP tour to solution


### TSP without constraints
Initially we were not quite sure whether it is fruitful to start with a TSP tour without considering configurations. However, as @cnumber pointed out, in this [thread](https://www.kaggle.com/competitions/santa-2022/discussion/370129), that the lower bound by minimum spanning tree is approximately 72599 and considering it is relatively loose as also mentioned by @elvenmonk, at [here](https://www.kaggle.com/competitions/santa-2022/discussion/376079#2086550). We were vaguely speculating that top score in LB starts with a TSP tour. 

We run LKH for a few days and it gave us 74077 solutions. We also run a few sub-problems with Cplex and hope it would give us a better solution. Although Cplex wasn't able to solve the problem because of its size, it was able to moving the lower bound very slowly which ends up tighter than the one obtained by spanning tree. We were convinced that we were probably on the right track given the gap between lower and upper bound.

We spent quite an amount of computation with LKH, the best solution is around 74075.3. Our secret sauce for a better TSP tour is [GA-EAX](https://pubsonline.informs.org/doi/10.1287/ijoc.1120.0506), which is also used in [1st](https://www.kaggle.com/competitions/santa-2022/discussion/379167) & [4th](https://www.kaggle.com/competitions/santa-2022/discussion/379080) place solutions. GA-EAX eventually gives us a **74074.95** tour.


### Delve into the restrictions
We suspected that one could convert TSP tour to solution for free once the first link successfully reached `(64, ±64)`, because the degree of freedom is plenty. To validated our hypothesis, we begun with a tour going straight up/down, i.e starting with
`(0, 0), (0, 1), (0, 2), ... (0, 64)`
and ending with 
`(0, -64), (0, -63), (0, -63) ... (0, 0)`.

We had a success with this tour with a few extra cost. This proof-of-concept experiment helps us to identify another restriction. (Our method of converting tour to solution is deferred to last section.)

1. Do not connect any two of (-127, -127), (-127, 127), (127, -127), (127, 127) with a straight line. (quoted from @cnumber and @kibuna's [solution](https://www.kaggle.com/competitions/santa-2022/discussion/379167))
2. Do not travel all four quadrants in a short span. (We later found this restriction but it was an easy fix)


### Manual choice of starting and ending subtour

Our best tour (blue tour in the background) turns left too early. We therefore manually perturbed a few edges (for example, one is circled in red). It ends up with two fixing sub-paths, the one in orange leaving the origin, and the one in purple arriving to the origin. These fixed paths mostly follow our best solution, with a few perturbations, but end up in good regions which allow the first link to successfully reach `(64, ±64)`.



To get these two paths, we locally optimize what to perturb within a 65x65 sub-image in the center. We eventually freeze these paths and re-run LKH and GA-EAX to get our final tour.


### Tour to solution

We formulate the tour to solution problem as Integer Programming (IP), to meet the arm moving constraints, and solved it with Cplex. It is impossible to solve the entire tour due to the size of configuration space. We thereby divided tour into smaller pieces with length 200. Sometime it is also infeasible to solve a 200-points piece, say the interval [K, K+200), because of a bad initial configuration at K. We gradually increase the interval, i.e. [K-100, K+200), [K-200, K+200) ..., until it is solved.

### Others we have tried or planed to
- We also considered the possibility of tours with repeated points rather than a Hamiltonian one. (didn't improve).
