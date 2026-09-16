# Winner solution

Competition: santa-gift-matching
Rank: #1
Source: https://www.kaggle.com/c/santa-gift-matching/discussion/47376

I'll go straight into my solution to keep this post as concise as possible. <br>
For simplicity, some things might be inaccurate or don't reflect what I actually did. <br>
code: <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution">github</a> <br>
<br>
<br>
<h1>Term</h1>
 \- Solution: A strategy or a sequence of algorithms to solve the ORIGINAL santa problem. <br>
 \- Assignment: A set of "which present will be given to which child". What we submit to kaggle is an assignment. I also occasionally compute assignments for the relaxation problem. <br>
<br>
<br>
<h1>TL;DR</h1>
\- I maximized (1,000,000 * ANCH + ANSH). <br>
\- My hunch tells, an assignment maximizing (1,000,000 * ANCH + ANSH) also maximizes ANCH ^ 3 + ANSH ^ 3. <br>
\- There are many almost-optimal assignments close to the optimal assignment for the relaxation problem. However, the actual optimal ones are located at a bit more distant places, though still close. <br>
\- I used depth first beam search. <br>
\-- This can compute a strictly optimal assignment. <br>
\-- Main node selection strategy is human eyeballing and hunch. <br>
\-- I used a min-cost flow algorithm to solve a relaxation problem. (time: 1 ~ 3 hours, code: <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/data_structure/min_cost_flow_dup_binary_heap.rs">min_cost_flow_dup_binary_heap.rs</a>) <br>
\-- I developed another min-cost flow algorithm to be used in beam search. <br>
\--- It solves a min-cost flow problem by repeatedly improving an initial flow. (time: 5 ~ inf seconds, code: <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/loop_solver.rs">loop_solver</a>, <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/loop_canceler.rs">loop_canceler</a>) <br>
<br>
<br>
<br>
<h1>Observations</h1>
<strong>1. I kind of feel that optimal assignments for (1,000,000 * ANCH + ANSH) are also optimal for ANCH ^ 3 + ANSH ^ 3.</strong> <br>
child_wishlist.csv is 100 times larger than gift_goodkids.csv.
I guess even a completely random assignment scores 100 times larger ANCH compared to ANSH.
We are maximizing cubics of them and ((ANCH + delta) ^ 3) - (ANCH ^ 3) ~= 3 * (ANCH ^ 2) * delta.
Therefore, increasing ANCH achieves 100 ^ 2 times better score gain than ANSH.
Actually, an optimal assignment has around 1,000 times higher ANCH. So, I optimized for (1,000 ^ 2) * ANCH + ANSH.
<strong>I'll talk about maximizing (1,000,000 * ANCH + ANSH) below</strong>
<br>
<br>
<strong>2. A relaxation problem which ignores twin/triple constraints can be formulated as a min-cost flow problem.</strong> <br>
You can check this <a href="https://www.kaggle.com/c/santa-gift-matching/discussion/45857">post</a> (Thank you, mtnn)
I'll call this problem "the relaxation problem" below.
<br>
<br>
<strong>2.1. Most of the twins/triplets in an optimal assignment for the relaxation problem follow the twin/triplets constraints.</strong> <br>
In the relaxation problem, costs of twins/triplets were averaged, and have the same values.
So, a pair of twins is likely to receive the same present (same for triplets).
As a result, for each present, except for some edge cases, there are at most one twin and one triplet which don't follow the twin/triplet constraints.
<br>
<br>
<strong>3. If we know how many singles, twins, and triples receive each present, the original santa problem can be formulated as a min-cost flow problem.</strong><br>
An example of this problem is like,<br>
\- we already know present 1 will be given to 990 singles, 4 twin, 6 triplets, ...., present 1,000 will be given to 986 singles, 2 twin, 12 triples.<br>
\- These numbers are valid (present 1 will never be given to 2 * n + 1 twin nor 3 * n + 2 triplets).<br>
Then, we can compute an optimal assignment by re-formulating this problem into a min-cost flow problem.<br>
<br>
I'll skip the detail of this formulation, but the point is that we can ignore the original twin/triplet constraints.
An optimal assignment for this problem automatically follow the original twin/triplet constraints because of what I said above (section 2.1).
The graph will have 3000 present nodes. Each present will have three node for single, twin, and triplets.
<br>
<br>
<strong>3.1. Here is an inefficient naive solution using the above formulation.</strong><br>
<pre>def compute_optimal_assignment_naive():
  scores = []
  for assignment_num_present_1_to_single in range(0, 1001):
    for assignment_num_present_1_to_twin in range(0, 1001):
      for assignment_num_present_1_to_triplet in range(0, 1001):
        .
        .
        for assignment_num_present_1000_to_single in range(0, 1001):
          for assignment_num_present_1000_to_twin in range(0, 1001):
            for assignment_num_present_1000_to_triplet in range(0, 1001):
              if are_these_assignments_valid(assignment_num_present_1_to_single, ....):
                scores.add(compute_optimal_assignment(assignment_num_present_1_to_single, ...))
  return np.max(scores)
</pre>
<br>
<strong>4. A fusion problem of the above problem and the relaxation problem can be formulated as a min-cost flow problem.</strong><br>
Here is an example of this problem.
We know present 343 will be given to 990 singles, 4 twins, and 6 triplets,
We also know present 532 will be given to 983 singles, 8 twins, and 9 triplets,
And, we don't know about the remaining.
We are allowed to ignore twin and triplet constraints for these 998 no-constraint presents.
<br>
This problem can be formulated as a min-cost flow problem.
I'll skip the detail of this, but we can simply combine the formulations for the above problem and the relaxation problem.
<br>
<br>
<strong>4.1. Here is a depth first beam search solution using the above formulation.</strong><br>
<pre>def compute_optimal_assignment_depth_first_beam_search():
  # Create beam search candidate queue.
  fusion_state_queues = [PriorityQueue() for _ in range(1000 + 1)]

  # Add initial state
  initial_fusion_state = create_1000_no_constraint_presents()
  initial_fusion_score = solve_min_cost_flow(initial_state)
  fusion_state_queues[0].push((initial_fusion_score, initial_fusion_state))

  final_scores = []
  while are_there_remaining_states(fusion_state_queues):

    for present_id in range(1000):
      state, score = fusion_state_queues[present_id].pop()
      if score &lt; np.max(final_scores):
        continue

      for assignment_num_to_single in range(1001):
        for assignment_num_to_twin in range(1001):
         for assignment_num_to_triplet in range(1001):
           next_state = current_state.clone()
           next_state.add_constraint(present_id, assignment_num_to_single, ...)
           next_score = solve_min_cost_flow(next_state)
           if nex_state.is_valid_state():
             fusion_state_queues[present_id + 1].push((next_score, next_state))
  return np.max(final_scores)
</pre>
<a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/beam_search.rs#L264">My actual code</a><br>
<br>
<br>
<br>
<h1>My solution</h1><br>
<br>
A. Compute an optimal assignment for the relaxation problem.<br>
B. Run depth first beam search for a limited space close to the assignment of the relaxation problem.<br>
B.. There is a fast min-cost flow algorithm only for this specific problem.<br>
C. Re-configure present priority for node selection based on my eyeballing impression during the previous beam search.<br>
D. Repeat 2 and 3 with broader search spaces.<br>
<br>
<br>
<strong>A. Compute an optimal assignment for the relaxation problem.</strong><br>
I implemented a min-cost flow algorithm. This is my very first program in rust.
This is an algorithm for a graph with non-negative cost using a concept named "potential". I'll skip its detail.
Each step in the algorithm, it's detecting multiple min-cost paths on a residual network and saturate their flows at once. <br>
Code: <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/data_structure/min_cost_flow_dup_binary_heap.rs">min_cost_flow_dup_binary_heap.rs</a>
<br>
<br>
<strong>B. Run depth first beam search for a limited space close to the assignment of the relaxation problem.</strong><br>
As I said "Most of twins and triplets in an optimal assignment for the relaxation problem follow twin, triplets constraints".
So, the search space is around the optimal assignment and its really small.
<br>
<br>
<strong>B.. There is a fast min-cost flow algorithm only for this specific problem.</strong><br>
An optimal min-cost flow must not have a negative cost cycle in its residual graph,
so we can solve a min-cost flow problem by repeatedly detecting negative cycle and cancel it, starting from some initial valid flow.
In depth first beam search, we add one constraint at a time. This means we know the optimal assignment before adding constraint.
We can create a nearly-optimal valid assignment from this previous optimal assignment by some heuristic.
This nearly-optimal assignment can be usually transformed into an optimal by cancelling a few negative cycles, so it doesn't take time.
We can use Bellman-Ford algorithm for negative cycle cancelling.
It takes O(|V||E|) = O(1,000,000 * (100 * 1,000,000)) for cancelling one cycle.
However, since our graph is almost bipartite graph, so we can pre-compute present-present distances beforehand, and detect a negative cycle in a present only graph.
This will reduce time complexity to O(|present| (|present| * |present|)) = O(1,000 * (1,000 * 1,000)).
Furthermore, in our problem, almost all of the negative cycles are short.
So, we can make this faster. I'll skip its detail. <br>
Code: <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/loop_solver.rs">loop_solver</a>, <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/loop_canceler.rs">loop_canceler</a><br>
<br>
<br>
<strong>C. Re-configure present priority for node selection based on my eyeballing impression during the previous beam search.</strong><br>
There are only several presents which affects score during beam search.
If we know which present would affect score, the beam search can be finished within the depth of 10 not full present 1000.
So, I run beam search with a really limited search space first, and listed up presents which affected score, then gave them priority. <br>
Code: <a href="https://github.com/ckomaki/kaggle-santa-2017-winner-solution/blob/master/src/rust/beam_search.rs">beam_search.rs</a><br>
<br>
<br>
<strong>D. Repeat B and C with broader search spaces.</strong><br>
Yeah, we can repeatedly increase the search space.
When we increase this search space to the infinite, the best score is proven to be optimal.
<br>
<br>
<br>
<br>
