# 1st place solution with visualized route

Competition: santa-2022
Rank: #1
Source: https://www.kaggle.com/c/santa-2022/discussion/379167

2023/01/19 added visualized route of the optimal(?) route
2023/01/23 made our [github repository](https://github.com/chettub/santa2022) public


We ( @kibuna and @cnumber) would like to thank the organizers for this annual optimization challenge and very interesting problem.

# Overview
This problem can be divided mainly into two steps. The first is to obtain the minimum cost path on the image with color cost and minimum reconfiguration cost as defined [here](https://www.kaggle.com/competitions/santa-2022/discussion/370129), and the second is to construct a valid arm configuration with minimal overhead costs. We worked on the former part by solving TSP with additional constraints using “genetic algorithm using edge assembly crossover (GA-EAX)”, and for the latter, beam search was used.
The two important points are
1. There exists impossible (or quite costly) routes due to the arm configuration that need to be removed when solving the TSP.
2. Once a reasonable path is obtained, it is relatively easy to obtain the arm configuration even if there are complicated moves, e.g. continuous long steps, because of the huge number of arm combinations which recover the same position.

# TSP with path-dependent constraints:
The basic approach to this problem is to search the minimum-cost path which starts from point (0, 0), goes around all pixels of the image, and returns to point (0, 0). This sounds like a standard TSP, but the arm constraint limits the possible routes. We incorporated this constraint with the normal TSP by adding a path-dependent cost; adding an additional cost when each of the constraints below are violated.
1. Do not enter the \\(x < 0\\) region until moving more than or equal to 64 steps in the y-axis direction.
    -  from the need to rotate the 64-arm at least 64 times to go into the \\(x < 0\\) area
2. \\( 2 \times x < \\) total y-axis direction moves
    - from the need to rotate the 32, 16, …, 1-arms
3. Do not connect any of the two of (-127, -127), (-127, 127), (127, -127), (127, 127) with a straight line
    - It can be confirmed that there is no valid arm configuration, unless adding additional reconfiguration costs. 
4. When the arm moves from (1, -4) to (0, -4) or from (1, -5) to (0, -5), and then through the 4th-, 3rd-, 2nd-, and 1st-quadrant of the image, the arm must move more than 128 times in the y-plus direction before going into the 1st-quadrant.
    - What we really wanted to do was to check if there were enough moves to rotate the 64-arm when the arm moves across the quadrants. However, this heavily increases the computational cost. We picked up the condition above after looking into typical paths generated with conditions 1, 2 and 3. In the end, these 4 conditions were enough to exclude undesired paths.

All of the constraints above have path dependency, therefore it is time consuming when incorporating them to a cost function in local search algorithms, because these constraints need to be checked every time a new candidate path is generated, which is one of the reasons why we chose genetic algorithm (GA). In terms of implementation, we didn’t exclude all the paths that violate the constraints, but instead imposed a penalty cost when the conditions are violated, so that the number of the violating paths in the population gradually reduce as GA progresses.
We have also utilized a visualizer (see Apendix 1) to consider the restrictions.

## GA-EAX-restart
[GA-EAX](https://pubsonline.informs.org/doi/10.1287/ijoc.1120.0506) was used for searching the minimum-cost path. The implementation is based on  [GA-EAX-restart](https://github.com/senshineL/GA-EAX-restart) with some modifications. The codes are available [here](https://github.com/chettub/santa2022) [chettub/santa2022].
GA-EAX is a GA-based TSP approximation algorithm using a fast and efficient crossover operator called edge assembly crossover (EAX), which is known to have updated the best known solutions for numerous TSP instances, especially for large [TSP instances of ~100,000 cities](https://www.math.uwaterloo.ca/tsp/data/ml/monalisa.html). We adopted this algorithm believing it had advantages in this competition due to the problem’s large size (66049 pixels), simplicity of implementing path-dependent restrictions, and the fact that the cost function is computed a relatively small number of times compared to local search algorithms.
We made several modifications to the original code. First, we added the constraints mentioned above. Second, we parallelized the algorithm to speed up the computation. Third, we made it possible to specify the initial population and also to autosave the population during the calculation. This allowed us to suspend and resume calculations, merge any populations we chose, calculate only the final stage of the optimization algorithm to improve solutions further, and utilize “restart” methods described in the [paper](https://pubsonline.informs.org/doi/10.1287/ijoc.1120.0506).
For higher accuracy cost calculation, the internal data type of the cost was modified to 64 bit integer (and to 128 bit integer later on). We have also modified the implementation of the distance function to retain only the necessary elements of the distance matrix for reducing memory usage, since the full-matrix distance was memory consuming.

# Arm Configuration
Once the minimum-cost tour was found, the corresponding arm configurations needed to be found. Here, we searched for the optimal arm movement without additional reconfiguration cost by using dynamic programming (DP) and beam search with pruning.

## DP
The configuration was most restricted by the 64-arm, so the first step was to determine whether the 64 arm could move along the given path. The 64-arm has 64*8 possible states. By considering the positions of the other arms, the following condition is required at each step.

$$ L^{\infty}(Position - 64ArmPosition) \leq 64 $$
where
$$ L^{\infty} = \max(|x|,|y|) $$

The above condition must be satisfied at every step. Also when considering only reconfiguration-cost optimal paths, the 64-arm can move only in the same direction as the path.

In the actual algorithm, we considered not only the 64-arm but also the 32-arm. Taking the condition above in to account, we calculated a 3-dimensional DP table ([step][32-arm configuration][64-arm configuration]) to check what 32-64-arm configuration was possible at each step. If none of the 32-64-arm configs were available at a step, the path was regarded as unrecoverable. The total size of the DP table was \\( 64 \times 8 \times 32 \times 8 \times 66049 \sim 9 \times 10^9 \\), and the number of state transitions from a 32-64-arm configuration was 9, therefore the entire DP table could be calculated in a few seconds when implemented with C++.

## Beam search
Beam search was used to construct the arm configuration. All possible arm configurations at each step were calculated from the configuration of the step before. The width of the beam search was 500. To speed up the search, we added some prioritization and pruning.

- Using the DP table above, we excluded configurations with invalid 64 and 32-arms.
- Configurations with more arms in the corner were selected with higher priority, as they have a higher degree of freedom.
- If no configurations were available, the search was rerun from several hundred steps before with doubled beam width.

Implementing the algorithms in C++ and adding various speedups allowed us to find an arm configuration for a given path in a few minutes. Also in the scope of our experiments, arm configurations were found for all paths that satisfied constraint 1~4.

# Experiments
The LB 1st place solution could sometimes be found in less than a day of computation with a 30 vCPU instance.
We have experimented with many more instances and have not yet found a better solution.

# Thoughts
- Perhaps GA is suited for path-dependent constraints because
 - GA-EAX-single is not too strong in local search?
 - of the population's diversity?
 - the number of cost calculation is relatively small, so it can perform relatively fast even when adding an additional time-consuming cost function.
 - it is stronger than LKH with large instances or when looking for global improvements?
In this problem, being able to save the population and apply GA again after adding additional constraints helped us very much.

# Appendix
## Appendix1. Visualizer
We used a visualizer to examine paths that the arm movement couldn’t be recovered, and discovered the 4th constraint.

## Appendix2. What did not work

### Consider non-optimal arm movements.
- The color cost improvement was not worth the reconfiguration cost. For example, if we move 3 arms when moving to an adjacent pixel, the reconfiguration cost increases from 1 to sqrt(3) (an increase of 0.7). However, even with a constraint violation penalty of 0.7, we have not found any route that surpasses the current best solution. Also, removing the 3rd constraint decreased the TSP cost by 0.19. This is not worth the additional reconfiguration cost.

### LKH
- In the beginning, we tried LKH for unconstrained TSP. However, adding constraints was difficult from implementation cost and calculation speed perspectives. We tried to make LKH output a route that satisfied conditions 1, 2, and 4 by determining the route around the origin, but the performance was poor and the score could only be reduced to near 74077. (We were wondering that perhaps the difference between the 74075.* and 74077.* teams were the difference in whether or not they manually chose the route around the origin.)
- One day of LKH calculation starting from the best GA-EAX solution found didn’t show any improvement.
- We considered redefining the problem as “Traveling salesman problem with time windows (TSPTW)”, but gave up due to the small city number and long run time of LKH's TSPTW benchmark.

### Concorde
- It was impractical to complete the calculation as it takes several hours to several days for a subgraph with about 5000 vertices. On the other hand, parallelized GA-EAX solved the same 5000-verticies TSP within a few minutes.

### GPX2
- We tried applying GPX2 to the set of discovered solutions for further improvement, only to find that most of the solutions found using GPX2 were violating the constraints. It might be because while GPX2 can efficiently find a local minima, it cannot take into account route-dependent constraints (at least with our implementation).

## Appendix3. What we didn’t have time to work on
- Allow routes that passes a pixel more than twice
  - Due to the convexity of the cost function, it is usually better to jump over a pixel than to go through it twice. Only places with long jumps where very large color costs are involved should be considered.
Visualizing the difference between two paths and highlighting “circuits” that could be improved, like the idea used in GA-EAX.
- Merging initial population with free energy minimization (ref. regularization of TDGA)
- LKH is reported to be stronger than GA-EAX for lattice-like TSP. In the later stages of the optimization, there were multiple paths with approximately the same cost, as in the case of lattice-like TSP. We wondered if the path could be further improved using LKH.

# One of our 74075.70654 Solutions


