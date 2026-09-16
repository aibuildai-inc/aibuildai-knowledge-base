# 13th Place Santa25: Billiard + customized Sparrow

Competition: santa-2025
Rank: #13
Source: https://www.kaggle.com/c/santa-2025/writeups/13th-place-santa25-billiard-customized-sparrow

# Overview
This year's Santa challenge was an exciting opportunity to learn about and develop new optimization strategies.  Thanks to my teammates @danielphalen @yuchen2066 @veniaminnelin - it took all our combined efforts to get this Gold medal.

Our basic approach combined several ideas:
1. Computing "optimal" regular patterns tiling 2-trees in a rectangular grid.
1. Modifying Sparrow to minimize square (vs. strip) scores
1. Implementing a C++ variant of billiard (described below), from paper by Gensane and Ryckelynck
1. Splicing together prior solutions

# Progression
We started out, as many others did, by computing the best regular patterns.
This was done by first computing the parameters of a 2-tree tiling for a coarse grid of angles and horizontal/vertical spacing.  Then these were optimized with Nelder-Mead or Powell for specific metrics.  Early on we just computed the best square packing parameters for whole rows and columns.  Later, when were combining patterns with extra trees, we could compute the best regular tiling (non-square) that had certain margins (say from .5 to 1.5) when combined with the extra trees needed.  These could then be fed to a modified sparrow with the fixed pattern as one big shape, and the extra trees as free polygons.
We next were able to run billiard against our best solutions repeatedly to get refined (slightly better) solutions but not substantially different in layout.  We needed more diversity to get to the next level.
Our final technique, which proved reasonably effective, but perhaps too late in the competition, was splicing together pieces of prior solutions, and then running combinations of sparrow and billiard on them. 

# Computing "Optimal" regular patterns
This was described above; we limited the search to perpendicular translations, but as @jeroencottaar has pointed out, you can do better with non-perpendicular patterns.  For a horizontal/vertical packing, with trees at an angle of 9.750144 degrees and 180 thereof, you can an area of .310709 per tree in an infinite grid.

# Modifying Sparrow

Similar to other teams, we modified sparrow using Claude or Codex, as we are unfamiliar with Rust.   We found that sparrow was good at working with about 50 objects at once, but then started to degrade rapidly in terms of finding improvements.  

1. Adjust the objective to be square packing instead of strip.
2. Allow warm start of solutions
3. We found swapping single trees was ineffective at exploring the parameter space, so we altered the disruptions during the explore phase to swap clumps of trees, move an edge of trees, or scramble a cluster of trees.

To adjust for the 50 object limit, we began using a merged block of trees as one large object and then packing many trees around that object.  In addition, we found the compression phase was not as efficient as we could get, and relied on the billiard algorithm described below, which usually would decrease the side length by about 0.2% over the results from sparrow.

# Billiard
In the referenced paper of Gensane and Ryckelynck, the WithPerturbations algorithm (what I am calling Billiard) tried to grow the maximum squares before collisions.  Converting that into fixed size squares that you try to pack into minimal area, we get an algorithm that looks like this (hyperparameters just for illustration):

```python
#Function WithPerturbations(C):
Start with a configuration C
Pick eps = .1, final_eps = 1e-7
while eps > final_eps:
    C0 = C
    Perturb(C, eps)
    Billiard(C)
    Squeeze(C)
    if C improved:
        accept C
        eps = eps*2
    else:
        reject C  (set C=C0)
        eps = eps/2
```
Perturb(C,eps) took 3 different forms:
1. perturb each tree by eps, then expand the solution so that there are no overlaps
2. simply expand the solution by 1+eps
3. expand the solution by 1+eps, then perturb each tree so there are no overlaps
[More on this later]
Billiard was pretty simple:
```
#Function Billiard
eps=start_eps
While (eps > start_eps/100):
    RandomWalking(C, eps)
    if improved score:  
        accept C
        eps=2*eps
    else:
        reject C
        eps=eps/2
```
RandomWalking was just randomly perturbing trees feasibly:
```
#Function RandomWalking
for i=range(Na):
    choose a random tree
    perturb it by eps; accept if it fits within current boundary and does not collide with existing trees
```
Finally Squeeze() was a bisection search to compress (scale centers of trees to origin) as much as possible without collision.

## Improvements
There were two modifications to the Billiard as described above that we used:
1. Instead of it operating on a single solution at a time, we allow it to accept and operate on a list of solutions, each loop of WithPertubations occuring in lockstep on each solution.  This allows us to start with a larger number of solutions, then gradually remove the worst scoring ones as the procedure progesses, ending up with 1 at the end.
2. With that list addition, we could now also add the ability to sometime add back in solutions that were not always the best - that is, instead of throwing away a worse solution, we could sometimes add it to the list and allow it be optimized for a bit. Sometimes that solution would pass the greedy solution and be the winner.

## Hooking in Expansion
We found that sometimes, you could not expand a configuration a small amount without collisions.  Upon investigation, this hooking phenomenon was observed.  Consider this 2 tree arrangement:


If you expand this configuration by just 1.001, you get this:



With a collision as indicated.  The reason for this is that you can think of an expansion of the tree centers is equivalent to an expansion of every point including tree vertices, followed by a shrinking of each tree to get it back to its original size.  The expansion does not cause any collision (its just a scaling), but the shrinking causes the branch of tree 0 to "hook" onto the branch of tree 1.

The way we fix this to note that if we move tree 0 forward by .001/2, the collision is relieved.  This is because that hooking branch is exactly 0.5 from the "center" or (0,0) point of the tree.  So we are able to achieve small expansions, of 1+eps, by doing this expansion, then repeatedly moving the offending trees forward by eps/2 or eps/4 until collisions are removed.  This is not fool-proof, but works in most cases.
Here is what it looks like with the tree0 moved forward to .001/2


# Splicing solutions
Our final descent into ~~madness~~ better score territory began with the realization that while billiard and sparrow were good at tweaking things, they were not so good at getting new solutions.  So we began generating new starting solutions from existing best solutions through randomly:
1. Splicing:  Take N and either (N-1 or N-2); rotate both randomly 0,90,180,270 degrees, then splice a random left vertical slice of N to a random right slice of the other, adding or removing enough trees to get to exactly N
2. Symmetrizing:  Take N and take a random left vertical slice, rotate it 180 degrees and merge it on the right side, again adjusting to get to N trees.
We then ran either Sparrow and Billiard or just Billiard on the new configuration, hoping for a better solution.
