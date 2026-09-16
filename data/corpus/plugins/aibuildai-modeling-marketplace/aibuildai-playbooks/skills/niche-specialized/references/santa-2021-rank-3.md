# 3rd place solution

Competition: santa-2021
Rank: #3
Source: https://www.kaggle.com/c/santa-2021/discussion/300509

First of all, I would like to thank the organizer for another year of interesting problems, and all the other participants for making the competition very fun. I would like to thank my teammates @nagiss, @kibuna and @hiroakikitahara for the great collaboration. 

In this post, rather than explaining the solution of 2428 directly, I would like to explain what we thought step by step. It may be a bit long, but I hope it will be helpful.

First step
---
Since dealing with 🌟 is complicated and we can only use at most two 🌟, we first worked on the solution without 🌟. If we fix which of the three strings each permutation appears in, we can solve the problem as an Asymmetric Travelling Salesperson Problem (ATSP). Since there are all 7! permutations and 2 * 5! additional mandatory permutations, if the permutations could be equally distributed among the three strings, the number of vertices per string would be approximately (7! + 5! * 2) / 3 = 1760. This 1760-vertex ATSP can be optimally solved in a realistic time using LKH and Concorde*. Thus, we only need to think about how to assign the permutations to the three strings.

*This means that it was true for this problem, not that it is true in general. For example, the superpermutation for n = 6 is a ATSP with 720 vertices, but it is considered difficult to find the optimal solution by Concorde in a straightforward way ([Houston 2014](https://arxiv.org/abs/1408.5108))

Assignment based on 2-cycle (Score: 2480)
---
First, we considered assigning the 5040 permutations in such a way that they contain the same number of mandatory permutations. Considering how other permutations should look like, it seems that permutations that can be moved at cost 1 (e.g. `7123456` -> `1234567`) should be in the same string as much as possible. Also, we want a group consisting of permutations that can be moved at cost 1 (e.g. `2345671` -> `3456712` -> ... -> `1234567`), and a group that can be moved from the group at cost 2 (e.g. `3456721` -> `4567213` -> … -> `1345672`. The cost from `1234567` to `3456721` is 2), to be in the same string. Repeating such a cost 2 group-to-group move n (= 7) times brings it back to the original group and forms a cycle. This cycle is called a [2-cycle](http://www.gregegan.net/SCIENCE/Superpermutations/Superpermutations.html#:~:text=method%3A%20Supermutate.js-,2%2Dcycle%20graphs,-A%20third%20way) in the context of the superpemutation (See the link for a more precise definition). We tried to use this 2-cycle to do the assignment.

There are 120 2-cycles, so 40 of them are assigned to each of the three strings. Each 2-cycle contains exactly one mandatory permutation, so at this point, every string will contain 40 mandatory permutations. By adding the 80 mandatory permutations that have not yet been added, we can satisfy the constraint in question.

The pseudo-Python code looks like this. After several iterations of solving the problem as an ATSP using LKH or Concorde with random 2-cycle assignment, a score of 2480 was achieved. We tried various methods of 2-cycle assignment, but could not find a solution with a score lower than 2480 without 2-cycle splitting described in the the following section.

```python
twocycle_nodes = []
for v in ["12" + "".join(s) for s in itertools.permutations("34567")]:
    nodes = []
    for y in range(6):
        nodes.append(v)
        v = v[2:] + v[1] + v[0]
        for x in range(6):
            nodes.append(v)
            v = v[1:] + v[0]
    assert v == twocycle
    twocycle_nodes.append(nodes)

assignment = np.arange(120) % 3
np.random.shuffle(assignment)

groups = [[], [], []]
for idx_groups, nodes in zip(assignment, twocycle_nodes):
    # TODO
    groups[idx_groups].extend(nodes)

    mandatory = nodes[0]
    assert mandatory.startswith("12")
    groups[(idx_groups + 1) % 3].append(mandatory)
    groups[(idx_groups + 2) % 3].append(mandatory)

solution = (solve_as_tsp(groups[0]), solve_as_tsp(groups[1]), solve_as_tsp(groups[2]))
```

Splitting 2-cycle (Score: 2440 w/o 🌟, 2430 w/ 🌟)
---

When we checked the solution for 2480, we found that many edges of length 7 are used to make the mandatory permutations appear that are not in the 40 2-cycles assigned to that string. For example, all permutations of the 2-cycle including `1234567` were assigned to only one string, so the two strings that were not assigned the 2-cycle were likely to use the edge with cost 7 to move `1234567`. To alleviate this problem, we thought of giving some of the permutations in the 2-cycle to the other strings. To alleviate this problem, we considered giving some of the permutations in the 2-cycle to the other strings. Specifically, we gave the permutations that can move to `12xxxxx` by using only the cost 1 edge to other strings. The splitting part of the pseudo code looks like this.

```python
groups = [[], [], []]
for idx_groups, nodes in zip(assignment, twocycle_nodes):
    mandatory = nodes[0]  # e.g. "1234567"
    neighbors = nodes[-6:]  # e.g. ['2345671', '3456712', '4567123', '5671234', '6712345', '7123456']
    nodes = nodes[1:-6]

    groups[idx_groups].extend([mandatory] + nodes)

    groups[(idx_groups + 1) % 3].extend([mandatory] + neighbors[:3])
    groups[(idx_groups + 2) % 3].extend([mandatory] + neighbors[3:])
```

We tried several assignments in this way and got 2440 solution without 🌟. Furthermore, by adding the following two ideas, we reached 2430 with 🌟.

 - For 2-cycle assignment, stop assigning randomly and create n = 7 2-cycles starting from n = 5 2-cycles and assign them in order.
 - Assuming that one 🌟 only shrinks one edge, randomly choose two mandatory permutations and replace the first 1 with 8 (🌟)


Non-TSP Solution by eijirou-san (Score: 2440 w/o 🌟, 2435 w/ 🌟)
---
Here, I would like to share the solution by @eijirou-san, who kept the first place on the Leaderboard in early stage of the competition. In this solution, the procedure of splitting 2-cycles is deeply considered, and the 2435 solution can be built without TSP solvers.

Consider a block constructed by deleting the permutations after the last 2-edge of a 2-cycle and connecting it with a 3-edge to a group brought from another 2-cycle (See the figure below). The total cost of this block is 56, but since it ends in `12`, it can be considered 54 due to the two-letter overlap with the next group. Furthermore, since one block contains two mandatory permutations, we can construct a solution with a score of 54 x 40 + 7 x 40 = 2440 solutions by adding the remaining 40 mandatory permutations at a cost 7.

[image]

We can construct the 2435 solution by inserting two 12xxxxx while utilizing two 🌟 (e.g. splitting 1672345 -> 7234561 into 1682345 -> 8234576 and 1278345 -> 7834561). However, we could not find a better solution than 2435 by this splitting method and 🌟.

The 2428 solution
---
We verified the lower bound in a similar way to the [one using TSP posted in Discussion]((https://www.kaggle.com/c/santa-2021/discussion/294139), and confirmed that this 2440 solution is optimal without 🌟, so we moved on to consider the construction of a solution using 🌟.

- As the 2440 w/o 🌟 method, Divide the 2-cycle into four blocks: S (1 mandatory permutation required), A (35 permutations), B (4 permutations), and C (2 permutations).
[image]
- In this case, one string should contain all 120 types of S and 40 of each of A, B, and C to achieve an even distribution.
- Due to the position of the "1" and "2", the minimum cost of the transition between each block is as follows (all transitions are required to be the minimum cost to achieve 2428)
  - S -> S ... 7 (Can move to a different 2-cycle)
  - S -> A ... 2 (Moves only within the same 2-cycle)
  - S -> B ... 1 (Moves only within the same 2-cycle)
  - S -> C ... 5 (Can move to a different 2-cycle)
  - A -> S ... 7 (Can move to a different 2-cycle)
  - A -> A ... 3 (Moves only within the same 2-cycle. Moving to the same block in the same 2-cycle does not make any sense.)
  - A -> B ... 2 (Moves only within the same 2-cycle)
  - A -> C ... 5 (Can move to a different 2-cycle)
  - B -> S ... 3 (Can move to a different 2-cycle)
  - B -> A ... 5 (More efficient to go through S)
  - B -> B ... 4 (More efficient to go through S)
  - B -> C ... 1 (Moves only within the same 2-cycle)
  - C -> S ... 1 (Moves only within the same 2-cycle)
  - C -> A ... (More efficient to go through S)
  - C -> B ... (More efficient to go through S)
  - C -> C ... (More efficient to go through S)
- The transition cost in the block is 38 for A, 3 for B, and 1 for C, for a total of 42
- We tried to make a unit of permutations (containing three S's and one A, B, and C, which can transition to another unit with minimum cost) using these. For example, if we make a unit such that A -> S -> C -> S -> B -> S -> (next A), the cost is 42 + 7 + 5 + 1 + 1 + 3 + 2 = 61, and if we connect 40 units of these, we get 61 x 40 = 2440. The structure of our 2440/2430 solution looked like this.
- Analyzing the 2430 solutions using 🌟 in a way that replaces two Ss with 82xxxxx, we find that while the cost of one A -> S is reduced from 7 to 1, the result is an improvement of -5 per 🌟, since we can no longer enter A at cost 2 elsewhere.
- The key to improving this and getting to -6 per 🌟 was to use 🌟 as more than two letters. We use 🌟 at the end of the A Block to reduce the cost of A -> S from 7 to 1 as shown in the figure below. A single 🌟 is used as three characters: 7, 6, and 1.
[image]
- In this usage, the last permutation of the A Block using 🌟 belongs to the 2-cycle of the transition destination S, so the last permutation of the original A Block has to be used somewhere. We note that if the A Block with 🌟 is from 1|2xyz**ab** (in the notation in [the Greg Egan's page](http://www.gregegan.net/SCIENCE/Superpermutations/Superpermutations.html#TCG)), then S is from 1|2xyz**ba**, and the two-cycles are related by a swap of the last two letters. Therefore, if we use 🌟 in the same way for the A Block of 1|2xyz**ba**, the last permutation will be swapped among the A Blocks using 🌟, and there will be no remaining permutations. We want A -> S to be a transition between different 2-cycles, so when we use 🌟 for this pairing, we make sure that each is assigned to a different string. In summary, the procedure is as follows.
   - Assign A Block of 1|2xyzab and 1|2xyzba to different strings
   - Swap the last permutation of the A Blocks between the two strings
   - Use 🌟 in the form 2xyza182xyzba and 2xyzb182xyzab respectively
- Using this, we can achieve -6 improvement per single 🌟, and reach the 2428 solution.

Honestly, we haven't prove the optimality of 2428. I tried to use Concorde for some relaxation problems, but I could not come up with the formulation of unconstrained use of 🌟.
