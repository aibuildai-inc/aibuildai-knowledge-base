# 11th Place Solution

Competition: santa-2025
Rank: #11
Source: https://www.kaggle.com/c/santa-2025/writeups/11th-place-solution

# Overview

My solution is based on Soft-Constraint Simulated Annealing combined with Lattice Potentials.

I managed the parallel execution using a Meta-Heuristic Framework (Tournament Selection) to generate elite solutions from a vast pool of candidates.
To further evolve the solutions, I applied a Grand Canonical Basin Hopping strategy, reheating initial solutions derived from N, N+1, and N-1 configurations.

Note: While some of these strategies draw inspiration from statistical physics, they are applied here as heuristics and may not be strictly rigorous from an academic perspective.

# Core Optimization Engine

This section details the logic of a single SA run. It serves as the fundamental execution unit within the Meta-Heuristic Framework.

## Base Solver : Soft-Constraint Simulated Annealing

I implemented a standard Simulated Annealing (SA) with soft constraints, allowing overlaps between particles and between particles and the container. Instead of decreasing the temperature, I annealed the system by increasing the penalty weight lambda.

Allowing overlaps is expected to make the search more efficient by enabling the solver to move between different local minima on the energy landscape more easily.

The container size is fixed, with an appropriate value assigned externally. In other words, instead of directly exploring the box size, the solver operates in a "Feasibility Mode," searching for whether a zero-overlap configuration exists for a given side length L.

## Post Polish Tool : NPT Ensemble

Since it is effectively difficult to completely eliminate overlaps using only the Base Solver, I developed a separate polishing tool to generate "legal" solutions for submission.

This tool uses an NPT ensemble with variable box sizes, incorporating volume expansion and contraction moves. The penalty lambda is gradually increased until all overlaps are eliminated.

## Lattice Potential

For large N, ordered structures are clearly superior. However, using a simple SA often results in disordered "glass states," regardless of the computation time. I solved this by introducing a lattice potential to the base solver to force particles into an ordered arrangement.

This lattice potential is turned off in the middle of the lambda schedule to allow the configuration to adapt to the container boundaries.

The lattice potential is based on a dimer lattice. I optimized the relative positions and angles of two particles, as well as the basic translation vectors, using SA to maximize the density in an infinite system. These pre-calculated parameters were then loaded and used by the Base Solver.

Additionally, for cases like N=156 where the particles fit neatly into the lattice grid, I separately generated initial solutions by annealing both the lattice parameters and the grid dimensions within a finite system.

# Meta-Heuristic Framework

A single SA run seems to plateau at around 70.0 due to the multimodal landscape and strong initial-solution dependency. To address this, I leveraged massive computational resources managed through a Meta-Heuristic Framework for efficient scaling.

## Tournament Selection

To efficiently explore the vast search space, I employed a massive parallel strategy. I ran up to thousands of SA instances simultaneously. The annealing schedule (increasing lambda) was divided into multiple generations (e.g., 50).

The population size was reduced exponentially at each generation, with survivors selected based on their penalty scores.

To maximize the diversity of starting points within a limited compute budget, I used the following strategy: Fewer iterations per agent were assigned in the early stages, which were then increased as the population decreased. This approach likely balanced the diversity of initial configurations with the overall search quality.

## Grand Canonical Basin Hopping

The best solutions were continuously refined using a strategy I call Grand Canonical Basin Hopping.

* Basin Hopping: To improve a configuration for N, the current best was used as an initial solution to restart the SA from a medium λ. This process was repeated in an infinite loop to escape local minima.

* Grand Canonical Approach: In the final stages, I utilized configurations from N+1 (removing one random particle) and N-1 (adding one random particle) as seeds for N. Inspired by the Grand Canonical Ensemble in physics, where particle numbers fluctuate, this strategy led to a significant score improvement from 69.30x to 69.10x.

**Diversity Maintenance (3-lineage method)**

Through the success of the Grand Canonical approach, I realized the critical importance of initial solution diversity. Since there was limited time to build a complex solution pool, a simplified "3-lineage method" was implemented to maintain diversity. Instead of a single best solution, the system tracked three separate lineages for each N:

1. The current best configuration for N.
2. The best configuration derived from N+1.
3. The best configuration derived from N-1.

New refinement runs randomly selected a seed from these three lineages.

# Other Methods & Trials

Below are some other methods I tried. They were not my main focus, but they were still effective in some cases.

## Replica Exchange

Replica Exchange is a common and powerful method in statistical physics, so I tried it early in the competition. I expected it to be effective for N <= 30 where ordered configurations are not always the best.

However, only a few results (like N=18) from this method survived in the end. It is hard to say if the issue was the parameter tuning or if the method simply didn't fit this problem well.

## Population Annealing

I tried a heuristic approach based on Population Annealing, using weighted resampling at each step. While it was effective in more cases than Replica Exchange, Tournament Selection generally performed better for this problem. It's unclear whether this was due to the implementation or the problem itself.

*Note: Tournament Selection was inspired by the logic of this approach.*

# Ideas for Further Improvement

To better maintain diversity in Grand Canonical Basin Hopping, the following ideas could be explored:

* Solution Pooling: Maintain a pool of sub-optimal solutions from the Tournament Selection (or, any method) and use them as seeds for initial configurations.

* Topological Diversity: Instead of just comparing scores, use a quantitative measure of topological differences to prioritize keeping diverse solutions in the pool.

* SA-style Pool Management: Instead of a hill-climbing approach (like the 3-lineage method), use SA-style acceptance for the pool. Varying temperatures across different pools could further promote diversity.

By evolving the entire environment of the solution pool rather than just the single best solution, it may be possible to achieve an "Open-Ended" improvement process—one that continues to scale its performance as more compute power is added, rather than hitting an early plateau.

# Appendix

## Compute Resources

I increased my compute resources as the competition progressed. The maximum configuration was as follows:

* 2× C4A (72 cores each) on Google Cloud Platform
* 2× Ampere A1 (80 cores each) on Oracle Cloud Infrastructure

Total Cost: < $2,000

## Generative AI Usage

I used Codex CLI to write the entire source code for this competition. Additionally, Gemini 3.0 Pro was used for brainstorming, generating ideas, and researching existing literature.

## Citation from Public Solution

While I primarily developed my own solutions, I found that merging public solutions in the final stages of the competition significantly improved my score (by approximately 0.01).

Affected N: 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, **14**, **131**, **132**, 154, **180**, **181**, **182**

The bolded values, in particular, represent essential improvements that my own solver was unable to reach.

## Statistics

| N range | Average score |
| --- | ---: |
| 1-20 | 0.400141 |
| 21-40 | 0.356848 |
| 41-60 | 0.348464 |
| 61-80 | 0.342586 |
| 81-100 | 0.339418 |
| 101-120 | 0.336823 |
| 121-140 | 0.334893 |
| 141-160 | 0.333553 |
| 161-180 | 0.331442 |
| 181-200 | 0.331146 |

**Total score : 69.106266**

I have attached a PDF file documenting all my final configurations (11thPlaceConfigurations.pdf). Please take a look if you are interested!
