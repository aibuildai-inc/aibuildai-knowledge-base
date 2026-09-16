# Santa25 14th place - Custom Simulated Annealing

Competition: santa-2025
Rank: #14
Source: https://www.kaggle.com/c/santa-2025/writeups/santa25-14th-place-custom-simulated-annealing

Hi all,

Huge thanks to the competition organizers and to all participants for making such a fun competition possible!

In a nutshell, my solution was:
1. Find the densest possible tree lattice (with a dimer or tree pair as basic unit).
2. Build many initial solution variations using that lattice.
3. Run a custom Simulated Annealing (SA) implementation to refine the edges, with fixed square side.
4. For solutions found in 3), do one of two things:
    - If the solution is valid, gradually compress it (reduce square side).
    - If the solution is invalid, gradually relax it (increase square side).

A video of the core SA algorithm running can be found [here](https://youtube.com/shorts/W1KTvhTJTfM).

Compute was a 16-core desktop to reach ~69.43 until the last week, then $200 in AWS ECS to ramp up to ~69.22 and snatch solo gold!

However, there are many implementation details that add up together to make it work, so I’m providing more details below in case anyone is interested.

Overall, it was a pleasure participating with you all! :)



### Solution details

#### 1) Densest lattice and differential evolution

- Densest lattice parameters were found using differential evolution. The lattice is a tiling of dimers (a dimer is composed of two trees, tree 1 and tree 2).
- Parameters or degrees of freedom are the 2 lattice basis vectors, tree 1 orientation and tree 2 relative NFP position, assuming an orientation of +180deg relative to tree 1.
- Objective to minimize is the area of the parallelogram defined by the basis vectors.
- Optimal lattice found has an area per tree of ~.3044.
- In the first weeks I tried using DE to directly obtain solutions for all N: added square side, lattice global orientation and lattice offset as variables, constraints on the number of trees inside the box, and changed objective to minimize square side. This got an easy ~71.x but was clearly not the best approach, although a solution obtained this way survived until the end! N=156 (.3299).

#### 2) Initial solution variations, freezing core trees, low to high N approach

- Identified 3 distinct separation lines in the optimal lattice, and aligned those with box walls following diverse criteria (eg. separation-aligned lattice aspect ratios, allowed wall overlaps, distances to walls).
- Defined a subshape of core trees to be marked as frozen when SA starts.
- Parameters for initial solutions are all a function of N, including the frozen subshapes’ parameters.
- Frozen subshape parameters also depend on other initialization parameters (eg. which separation line is being used).
- Broadly, for higher N more trees are frozen and for a longer duration, wall aligning biases are stronger, culling criteria is more rigorous, and lower random jittering is applied.
- For lower N wall biases are relaxed, the frozen core is smaller or non-existent, and initial solution parameters are relaxed.
- Given more time I’d have explored using different optimal lattices as basis for all this, since small tweaks that would lose on infinite plane density could make up by allowing different aspect ratios.

#### 3) SA custom algorithm

- Consists of two sequential tournaments: a calibration tournament, for finding the most promising initial solution configurations, and a main tournament using those winning initial configurations as starting points.
- Each tournament has multiple rounds where solutions are decimated until we end up with 16 finalists.
- Energy function is a weighted sum of inter-tree area overlap (quadratic behavior) and wall penetration depth (linear behavior). Quadratic behavior was found to reduce severe inter-tree jams, and achieved similar results and speed as inter-tree penetration squared. Wall collisions were encouraged to zero by the linear loss and a higher (but not infinite!) weight for the wall term.
- Temperature schedule follows 3 phases:
    1. High temperature phase, exponential decay (~10% of time)
    2. Critical phase, linear decay (sweetspot where we spend ~80% of time)
    3. Low temperature refine phase, resume exponential decay (~10% of time)
- Critical phase temperature bounds are identified by visual inspection of SA run videos, as merely watching energy levels proved unintuitive. Qualitatively a critical temperature is such that trees occasionally snap and change solution topology, yet are not all over the place.
- Moves or state transitions are standard translation/rotation local perturbations (gaussian), and global translation/rotation moves applied to all trees simultaneously. Initial solution parameters influence the magnitude of global moves.
- Crucially, local moves do not apply to frozen trees. Trees unfreeze after a while and can move locally again. This massively increased SA efficiency for large N.
- Global moves proved more effective to resolve long range imbalances compared to teleport or “flow along the outer shell” moves.
- Code is in python (I know :O) but heavily optimized with numba/fastmath. Some optimizations included:
    - Spatial hashing for linear time collisions.
    - Bounding box fast rejections before expensive polygon math.
    - Tree decomposition into 4 convex subshapes for fast SAT collisions.
    - Code unrolling in many places to avoid loop overheads.
    - Stop intersection tests early upon finding a gap.
    - Incremental energy updates for single tree moves.
    - Flat contiguous numpy arrays to maximize CPU cache locality.
    - Vectorized rotation matrices to avoid redundant math.

#### 4) Compressing and relaxing SA outputs

- Input solutions for this step were the finalists from the various runs of the main SA tournaments for different N and square side pairs.
- A simple faster variant of the above SA algorithm was used (with lower temperature and smaller moves) to compress all valid solutions and relax non-valid solutions.


#### Things that didn’t work

- Handcrafted solutions using a sophisticated GUI.
- JAX GPU raster-based differential evolution, where when discretized too much memory would blow up, and too little would not have the precision required for this problem.
- Multi-lattice initializations: in hindsight, it was clear this was the winning strategy, but alas could not achieve consistently lower scores compared to my main approach.
- Many others, this was a tough nut to crack...

Thanks for reading! :)
