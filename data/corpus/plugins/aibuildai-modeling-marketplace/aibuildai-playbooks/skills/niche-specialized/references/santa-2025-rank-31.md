# East Team Solution

Competition: santa-2025
Rank: #31
Source: https://www.kaggle.com/c/santa-2025/writeups/east-team-solution

# Woosung's Part (Memetic Algorithm)  
(I’ll update the other team members’ parts as well when I have time.)

In this competition, I investigated a hybrid approach combining a Genetic Algorithm and Simulated Annealing.


## 1. Initial Seed

[diagonal]

I created a symmetric initial seed by repeating a high-density 2-gram block pattern.

After rotating it by 45 degrees, I grew the (x, y) coordinates radially from the center in a circular manner.


## 2. Mutation

[diagonal_2]


### Corner Mutation

[step_1]

I created an operation that perturbs the corner regions during mutation.

Based on the AABB rectangle that encloses the entire polygon,   
I randomly select a corner region, remove that part, and then fill it in again.

Such mutations are applied to the elite individuals of the current generation and carried over to the next generation.   
We used operations such as flipping the orientation of a single corner, swapping two corners, or performing a three-corner swap.


## 3. Neighbor

The neighbor operations made only very small, fine-grained changes.

We mainly used 

- (1) Gaussian jitter
- (2) Vertical flip (V-flip)
- (3) Pivot rotation
- (4) Group rotation.


## (3) Environment 

[soft_wall]

We used two environments. Soft Wall and Hard Wall.

### (1) Soft wall (with rigid rotation)

[rot]

In the soft-wall setting, rigid walls are placed on the left and bottom sides, while the top and right regions are left free so that movement can occur freely under simulated annealing. We rotate the puzzle by 90°, 180°, and 270° so the walls work the same on every side. We do these rigid rotations very often—like rolling a small ball of dough.

### (2) Hard wall

In the hard-wall setting, the sides are fixed, and we only resolve collisions.

In Hard-wall, we borrowed Sparrow’s coordinate descent method. Since the shapes are simple, we used the true overlap area plus penetration depth directly, instead of relying on a proxy. Rather than pruning with an upper bound, we used a Genetic Algorithm in a way that functions like a stochastic beam search.


## 4. GPU / SIMD acceleration

[3_tri_1_quad]

We simplified the tree into three triangles and one quadrilateral (the quadrilateral uses only three non-overlapping axes).

This shape simplification allowed us to perform large-scale parallel processing using GPU/SIMD.


## 5. Conclusion

At first, I treated it as an initial-solution design problem,   
so I focused on seed growth rules and tried many greedy algorithms.   
I wish I had used the dense grid more.

Most solutions were found by chance after a lot of computation.

In the final phase, I focused on GPU-based parallelization.   
However, increasing the population size gave only logarithmic-like gains, so large-scale GPU parallel experiments didn’t have much impact.

---

# Amon's Part

Overall, I explored SA-based optimization and initial solution generation using tessellated patterns.

## Early Stage (start ~ mid Dec)

My main focus was to write a base Simulated Annealing (SA) code, observe the results, and identify bottlenecks. The major difference from the public SA notebook was that I implemented a logic to gradually decrease the magnitude of perturbations.

Specifically, I scaled the probability of different transition types based on the current temperature.
- **High Temperature:** The solver allowed large movements and structural changes, such as vertical flips and random tree rotation, to explore the global space.
- **Low Temperature:** The solver prioritized small adjustments and "compaction" moves (shifting trees toward the center), which increased the acceptance ratio even in later iterations.

Since neither the public nor my own tessellated structures were strong at this stage, this SA optimization starting from random tree placement outperformed the patterned solutions up to N ≈ 120.

## Mid Stage (mid Dec ~ early Jan)

Our team's focus moved to improving and mixing currently available solutions rather than creating them from scratch. I implemented a logic to create an initial solution for N trees by removing k trees from an (N+k)-tree solution. I mainly used k = 1, 2.

This allowed good local patterns found in larger N solutions to "flow into" smaller N solutions. The probability of a tree being removed was weighted based on its distance from the center (trees nearer the edges were more likely to be removed). This heuristic worked because it is generally easier to close gaps created near the boundary of the square than those in the center.

Combined with the optimization heuristics created by teammates, this approach improved our score in the mid-stage.

## Final Stage (early Jan ~ end)

By this stage, tessellated solutions had consistently outperformed random placements for larger N. I committed to exploring tight tree tessellation and generating a large number of "candidate" initial solutions by cropping square windows from an infinite grid canvas. This was a two-stage pipeline:

### 1. Grid Canvas Creation

The purpose of this stage is to create a tightly packed canvas of trees in a tessellated pattern. I explored three patterns: **REGULAR** (vertically stacked), **HEXAGONAL** (odd-row offset), and **SHIFT** (cumulative row offset).

I solved this by formulating it as a 6-dimensional optimization problem maximizing density. The parameters were:
- `angle`: The base rotation of the tree.
- `intra_dx`, `intra_dy`: The internal spacing between the two trees that form a single "module".
- `inter_dx`, `inter_dy`: The spacing between adjacent modules in the grid.
- `row_offset`: The horizontal shift applied to rows (used for HEXAGONAL and SHIFT patterns).

I optimized these using a nested approach: iterating over angles and module configurations, then applying binary search to find the tightest valid spacing for `inter_dx` and `inter_dy`. This yielded many tight grid patterns across various angles.

### 2. Initial Solution Cropping

The purpose of this stage is to create valid initial solutions for a specific N by cropping the infinite grid.

I implemented a simple SA algorithm to optimize the cropping window defined by `(center_x, center_y, side_length, angle)`. The objective function was the actual density of the cropped selection. We kept the top-k dense solutions for each grid pattern. This allowed us to technically create an infinite number of initial solutions that were guaranteed to have a good tessellated structure.


[grid optimizer]

Even though I was able to create many candidates, most did not end up achieving a better score than our existing best solutions, simply because those solutions had already undergone weeks of optimization. My regret is that I was not able to implement a robust method to "embed" these rigid tessellations into the center of the square and effectively fill the remaining boundary usage with random placement.
