# 7th Place Solution SA SA SA

Competition: santa-2024
Rank: #7
Source: https://www.kaggle.com/c/santa-2024/discussion/561091

Thank you kaggle host for designing the wonderful competition questions. I had a great time playing.

I haven't played Santa competitions in depth before, so I found a book called Essentials of Metaheuristics by Sean Luke in the early stages of the competition. After reading it, it inspired me a lot. Based on the knowledge in this book, I have designed the following solution. I will describe the design process of my solution from simple to complex, hoping to help newbees like me.

---

**TL;DR** Combining **Iterated Local Search** (ILS) with **Simulated Annealing** (SA) and leveraging pruned distance matrices, caching mechanisms, and distributed parallel computing to efficiently evaluate solutions for discrete sequence ordering problems.

---

### **Simulated Annealing Search (SA Search)**

I begin with a fundamental simulated annealing (SA) approach, which consists of three key components:

- Temperature decreases with each step.
- The current solution undergoes tweak.
- If the tweaked solution is better, it is accepted; otherwise, it is accepted with a certain probability.

```
# 1. Initialization
# 2. While the temperature is above the stopping threshold:
    # 2.1 Generate a neighboring solution: neighbor_solution = tweak(cur_solution)
    # 2.2 If the neighbor is better, accept it.
    # 2.3 If the neighbor is worse, accept it with a probability; 
    #       otherwise, keep the current solution.
  
    # 2.4 Reduce the temperature.
```

#### **Design of Tweak Operations**

- I designed multiple **neighborhood operations (tweak operations)**, including:

  - **Insertion**: Inserts a contiguous subsequence of length \(m_1\) into another position.
  - **Swap**: Selects and swaps two subsequences of length \(m_2\).
  - **Local Shuffle**: Randomly shuffles a segment of length \(m_3\).

- To efficiently generate all possible perturbations, I precompute all indices of potential perturbations and use NumPy indexing to retrieve all possibilities at once.

  - When \(m_1, m_2 = 1~20\) and \(m_3 = 1~5\), each tweak generates around 500K candidate solutions.
  - When \(m_1, m_2 = 1~65\) and \(m_3 = 1~5\), each tweak generates around 1M candidate solutions.

- These neighborhood operations create a vast number of permutations. In every step, I use batch processing to infer and evaluate \(k\) random candidates at once (where \(k\) ranges from 1000 to 5000).

#### **Batch Neighborhood Search**

Considering the computational advantages of matrix operations in large language models, I aim to maximize inference efficiency by performing batch evaluations instead of processing one sentence at a time in a loop. Thus, instead of tweaking a single neighbor and evaluating it, I tweak a large batch of neighbors and select the best one.

- If the best neighbor achieves a better score, it is accepted.
- Otherwise, a suboptimal solution is accepted with a probability to introduce randomness and escape local optima. If accepted, a random solution is selected from the top 10 candidates.

The SA search pseudo-code is now as follows:

```
# 1. Initialization
# 2. While the temperature is above the stopping threshold:
    # 2.1 Generate a batch of neighboring solutions: neighbor_solutions = tweak(cur_solution)
    # 2.2 Perform batch inference: neighbor_scores = batch_infer(neighbor_solutions)
    # 2.3 If a neighbor is better, accept the best solution.
    # 2.4 If no neighbor is better, probabilistically accept a suboptimal solution from the top 10.
    # 2.5 Reduce the temperature.
```

---

### **Multi-round Simulated Annealing Search (Repeat SA Search)**

Given that the neighborhood space for a 100-word sequence is vast, randomly sampling \(k\) candidates may still miss better neighbors. Thus, I dynamically adjust the search granularity by iteratively applying SA, increasing the number of attempts per temperature step for a finer search.

To determine when to stop repeating SA searches, I introduce an **early-stopping patience mechanism**. Initially, patience is high, allowing up to three consecutive unsuccessful searches. Later, patience decreases, and in the later stages, a single unsuccessful search terminates the process.

The repeat SA search pseudo-code is as follows:

```
# 1. Initialize patience and initial solution
# 2. While patience is not exhausted:
    # 2.1 Initialize the starting temperature
    # 2.2 While the temperature is above the stopping threshold:
        # 2.2.1 Perform multiple steps per temperature step:
            # 2.2.3 Generate a batch of neighboring solutions.
            # 2.2.4 Perform batch inference.
            # 2.2.5 Accept the best solution if it's better.
            # 2.2.6 Otherwise, probabilistically accept a suboptimal solution from the top 10.
            # 2.2.7 Reduce the temperature.
    # 2.3 Record unsuccessful attempts.
    # 2.4 Increase steps per temperature.
```

---

### **Iterated Local Search (ILS)**

Repeat SA search tends to get trapped in local optima, necessitating a **large perturbation** to escape. However, excessive perturbation reduces the search to random exploration, while insufficient perturbation fails to escape local minima.&#x20;

The perturbation strategy I adopted involves randomly selecting **perturb\_rank** positions and shuffling the words in those positions.

I set an initial **perturb\_rank** of 35. If the current local search yields a better result than the previous best, I scale **perturb\_rank** by a factor and continue searching.

The ILS pseudo-code is as follows:

```
Initialize parameters
while True:
    cur_sent = perturb(cur_sent, perturb_rank)
    local_bst = local search around cur_sent (aka. repeat sa search) 

    if local_bst < global_bst:
        perturb_rank *= 2/3
        Update cur_sent and global_bst
    elif local_bst < prev_local_bst:
        perturb_rank *= 5/6
        Update cur_sent
    elif local_bst >= prev_local_bst:
        perturb_rank = 35
```

---


### **Efficient Score Caching**

To avoid redundant perplexity calculations, I use a **score cache** implemented as a dictionary, where keys are sentence representations and values are perplexity scores.

To optimize storage efficiency, I:

- Encode words as integers.
- Convert integer lists into compact binary representations.
- Store them as binary arrays, significantly reducing memory usage.

To enable resumable storage operations, disk access  was necessary. To further accelerate disk read and load speeds, I leveraged the **msgpack** package, which significantly improved performance.

This approach reduces the storage size of a sentence representation from **685 bytes** (string) to **121 bytes** (binary encoding), enhancing caching performance.
  
---
### Pruning
#### **Using a Distance Matrix for Pruning**

For large-scale ordering within a neighborhood, the computational cost is enormous. If there is an efficient way to prune the candidate solutions in the neighborhood beforehand, the search efficiency can be dramatically improved.

Inspired by a naive understanding of language models and taking a simple 2-gram model as a cue, we assume that a sentence’s final perplexity is related to all of its 2-word combinations. This assumption allows us to transform the perplexity calculation into a Traveling Salesman Problem (TSP)-style distance calculation. We represent the distances between “cities” (i.e., words) using a distance matrix. For example, consider a text (text5) that consists of 100 words, represented by indices 0–99; with the tokens `</b>` and `</s>` represented by 100 and 101, respectively, the resulting distance matrix is of size 102×102. *(Note: Since some words are repeated, some indices were omitted in the actual implementation details.)*

However, the initialization and dynamic evolution of the distance matrix itself poses a new challenge. I adopted a very simple approach:
- **Recording Local Optima:** Each time the algorithm becomes stuck in a local optimum, I record that solution, manually maintaining roughly 20 local optima.
- **Matrix Initialization:** I create a blank distance matrix and initialize the cells corresponding to the 2-gram combinations present in these local optimum solutions with a relatively low value (e.g., 47). For edges that have never appeared in any historical local optimum solution, I initialize their corresponding matrix entries with a higher value (e.g., 100).

Using NumPy and this distance matrix, it is possible to sort and prune 1 million candidate solutions within a few hundred milliseconds—roughly improving the search efficiency by a factor of 100.

```python
# self.paths[:, 0] = 100, self.paths[:, -1] = 101
self.paths[:, 1:-1] = ng.gen_all_neib(current)  
distances = np.sum(self.distance_matrix[self.paths[:, :-1], self.paths[:, 1:]], axis=1)  
sorted_idx = np.argsort(distances)

seled_idx = np.concatenate([
    sorted_idx[:int(5 * k / 6)],
    np.random.choice(
        sorted_idx[int(5 * k / 6):], k // 6, replace=False)
])  
neighbors = np.unique(self.paths[:, 1:-1][seled_idx], axis=0).tolist()
```

---

#### **Dynamically Optimizing the Distance Matrix**

The manually maintained distance matrix described above can easily become biased toward historical local optima. Ideally, the distance matrix should be updated and optimized dynamically. During the search process, for solutions with perplexity below a certain threshold (for example, 33, 32, or 31), we extract the edges contained in these solutions and then update the corresponding values in the distance matrix. In the end, each cell in the distance matrix holds the average perplexity of all solutions that include that particular edge.

```python
if min(uncached_scores) < int(self.globe_bst[0] + 2):  
    old = (self.distance_matrix <= int(self.globe_bst[0] + 2)).sum()  

    min_neighbors = [
        l1 for l1, l2 in zip(uncached_neighbors, uncached_scores)
        if l2 < int(self.globe_bst[0] + 2)
    ]  
    min_scores = [
        l1 for l1 in uncached_scores
        if l1 < int(self.globe_bst[0] + 2)
    ]  
    bst_miniseqs = [[100] + nei + [101] for nei in min_neighbors]  
    bst_miniedges = [extract_edges(miniseq) for miniseq in bst_miniseqs]  

    count_arr = np.ones(101 * len(bst_miniedges))  
    score_arr = np.repeat(np.array(min_scores), 101)  
    edges_arr = np.array(sum(bst_miniedges, []))  
    unique_edges, inverse_indices = np.unique(edges_arr, return_inverse=True, axis=0)  
    grouped_count = np.bincount(inverse_indices, weights=count_arr)  
    grouped_score = np.bincount(inverse_indices, weights=score_arr)  

    n0, n1 = zip(*unique_edges)  
    self.distance_matrix[n0, n1] = (
        self.distance_matrix[n0, n1] * self.count_matrix[n0, n1] + grouped_score
    ) / (self.count_matrix[n0, n1] + grouped_count)  
    self.count_matrix[n0, n1] = self.count_matrix[n0, n1] + grouped_count
```

I recognize that this section may be somewhat challenging to understand. A more refined approach would be to adjust the distance matrix based on the perplexity of each specific edge rather than using the aggregate perplexity of the entire solution to update the matrix. I originally planned to improve this in a subsequent version, but fortunately, this version already achieved a final perplexity of **28.5**.

---

I've provided a high-level overview of my approach, but there might be some additional implementation details that I haven't discussed. If you're curious, my [source code](https://www.kaggle.com/code/max2020/santa24-7th-place-solution) is available.
