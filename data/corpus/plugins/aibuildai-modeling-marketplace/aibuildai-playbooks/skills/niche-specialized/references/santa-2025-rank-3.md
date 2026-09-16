# Third place solution: a customized sparrow algorithm

Competition: santa-2025
Rank: #3
Source: https://www.kaggle.com/c/santa-2025/writeups/third-place-solution-a-customized-sparrow-algorit

# Preface
First, I would like to extend a huge thank you to the kaggle team organizing this competition. This has been one of the most fun Santa competitions we ever participated in. Further, a huge congratulations to all participants and @jeroencottaar for the solo victory. Finally I'd like to thank all my fellow team mates without whom this achievement would not have been possible @zidmie, @khahuras, @delporterobbe and @olehugnivenko 

#  Introduction
A schematic overview of our solution can be found below. It consists of the following main components, which we will discuss more in-depth subsequently:
- Rectangle combination to reach good square solutions for large N from smaller rectangular solutions
- A heavily modified version of Sparrow that supports warm starts, directly optimizes the square objective, increases numerical precision, and introduces several new hyperparameters to control optimization behavior (Rust)
- A very fast C++ simulated annealing (SA) and “push” procedure to slightly perturb and compress square solutions (final step for each N)
- A seamless workflow to fully exploit teamwork and distributed compute resources



# Rectangle combination

Early in the competition, we achieved strong results by combining optimized rectangles. We generated a large library of rectangle bands with diverse aspect ratios using lattice generation and the vanilla Sparrow algorithm with different strip heights.

Rectangles from this library were then recursively combined to form larger rectangles and squares. While none of our final results were generated directly by this algorithm, these solutions served as valuable seeds for subsequent steps.

Below is an example of an output of the rectangle combination algorithm (with some post-processing already applied). Note that the rectangles in the green & blue zone are, in turn, again combinations of other rectangles.



As a funny coincidence, if we plotted out the minimum and maximum dimensions of our generated rectangles, birds (sparrows?) seem to appear!



By applying purely this algorithm, **a score of roughly 70.0** can be achieved.

# Simulated annealing + compression
Early on, we translated one of the SA notebooks that was [made publicly available in the first days of the competition](https://www.kaggle.com/code/crodoc/87-38-hybrid-optimization-strategy) into C++ and added multi-threading to it. Oleh, who only merged with us in January 2026 was using [another public notebook SA algorithm](https://www.kaggle.com/code/blueshyy/adaptive-sa) translated to C++. 

An outline of the SA algorithm is provided in pseudo-code below:
```
Initialize current layout and best layout
Compute initial bounding square size
Set temperature T = T_start

For many iterations:
  Pick one random tree
  Propose a small random move (dx, dy) and rotation (dθ)
  If the moved tree overlaps others: reject

  Compute new bounding square size (objective)
  Δ = new_score − current_score

  If Δ < 0:
    accept move
  Else:
    accept move with probability exp(−Δ / T)

  If accepted:
    update current layout
    update best layout if improved

  Gradually decrease T
```

In addition, we implemented a simple but initially effective post-processing algorithm that simulates shaking a tray containing the polygons. This was also implemented in C++.

An outline of this pushing algorithm is provided in pseudo-code below:
```
Repeat until no improvement:
  For each predefined push direction (e.g. left+down, right+up):
    For decreasing step sizes:
      For each tree:
        While tree can move slightly in push direction (and rotate a bit)
          without collisions or boundary violations:
            accept the move

    If resulting layout has smaller bounding square:
      accept it
```

By combining the rectangle combiner with these algorithms (to post-process both the rectangles in our library as well as the intermediate & final solutions of the combiner), a **score of roughly 69.50** can be achieved.

# Extending the sparrow algorithm
While we started using the “vanilla” implementation of the sparrow (credit to @jern97 and @tonywauters) algorithm quite early on to create rectangles for our rectangle library, we never fully explored it in depth. In January, Oleh joined our team and he had mostly been working on extending that implementation to **build in several inductive biases tailored to our specific problem**.

## Directly optimizing the square
Sparrow’s objective was modified to minimize both width and height simultaneously, instead of minimizing width while keeping height fixed. Aligning Sparrow’s objective with the competition metric was a natural first extension.

## Allow to pass along an initial solution for sparrow to start from (warm start)
We extended Sparrow to accept an initial solution, allowing it to warm-start from previously obtained layouts. This enabled us to seed Sparrow with the diverse solutions produced by the rectangle combination algorithm.

## Enforce symmetry in the solver (4-way and 2-way symmetry)
To significantly reduce the search space, we enforced symmetry constraints. Both 2-way and 4-way symmetry were supported for values of N divisible by 2 and 4, respectively. While few symmetries remain in the final solutions, this approach helped Sparrow converge to strong seed solutions early in optimization.

## Orientation biases to enforce lattice-like structures
To reduce rotational noise and stabilize global structure, we apply a soft orientation bias that encourages items to align with a dominant layout axis. For a placement with rotation θ, the penalty is



where w is `orientation_bias_weight` (hyper-parameter), θ* is a fixed target `orientation_bias_angle` (hyper-parameter), and Δ is the smallest angular difference modulo 180°.

The penalty is made center-aware by setting B(c) equal to



where r is the normalized distance to the container center, b = `orientation_bias_center_boost` (hyper-parameter), and p = `orientation_bias_center_power` (hyper-parameter), enforcing stronger alignment in the center and softer constraints near edges. The bias strength adapts during optimization (growing on improvement, decaying otherwise, and rescaled at compression start).

## Generating and kicking seed solutions
Additional to the seeds generated by our rectangle combination algorithm, we also implemented down- and up-sampling routines to generate initial Sparrow solutions from solutions at other values of N. Below is a simple example of how to up sample. We basically add a tree randomly and let sparrow insert it intelligently.



Much of the effort in the later stages of the competition was spent tuning these hyperparameters and carefully selecting both the target N and the N values to down- or up-sample from, often guided by visual inspection.

Before running Sparrow, initial solutions were “kicked” to help escape local optima by applying global rotations (rotating the entire solution by a few degrees) and scaling (multiplying all (x, y) coordinates by a factor slightly larger than 1).

# Teamwork

Ideation and balanced workload distribution are major challenges in Kaggle team settings, especially in a competition that demands relentless optimization to fully exploit available compute resources. Taking into account different backgrounds and time zones, we adopted a simple and efficient collaboration scheme:
- Kaggle notebooks (with internet enabled), automatically submitting each newly found improvement for a single N as one submission (thanks to Kaggle for allowing 100 submissions per day)
- Reporting scores and parameters directly to Slack from those notebooks for further analysis and discussion

Below an example of messages posted by our slackbot when improvements were found:


With this workflow, all team members could continuously run multiple notebooks 24/7 with minimal effort, alongside their local machines. Merging improvements and distributing new ideas was fast and straightforward.

# Extra figures (varia)

A graph with the score and side lengths per n can be found below:


The median angles for each of the solutions seem to show a clear pattern. Optimal solutions tend to have a lattice structure of trees with around 22-24 degrees and 202-204 degrees.


# LLMs
It would be dishonest not to mention the use of AI tools in this competition. While all core ideas originated from the team (otherwise a gold medal would not be possible), AI tools proved valuable for supporting tasks. In particular, they helped translate Python prototypes into efficient C++ implementations and assisted in modifying the Sparrow Rust codebase (a language unfamiliar to most of our team) making the extensions described above significantly easier to implement.

# Our solutions

Enjoy this video of our final solutions: https://www.youtube.com/shorts/Ru8dcAfGb-E

