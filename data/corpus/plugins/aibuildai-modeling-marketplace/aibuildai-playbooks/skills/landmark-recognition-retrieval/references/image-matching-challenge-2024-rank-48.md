# 47th Place Solution: A Novel Transparent Scene Trick!

Competition: image-matching-challenge-2024
Rank: #48
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/511507

Thank you to the organizers for holding this competition; we have learned a lot from it!

Our approach to handling categories other than glass cups is not much different from before; We made a mistake which led to overfitting of the non-glass cup categories on the public leaderboard **[LB: 11, PB: 47]**, **but** our technique for handling glass cup categories is novel and has improved a lot. below is the flowchart of our method.



**Transparent Scene Trick**

Through experimentation, we have discovered that the key to constructing a **rotational matrix** is to determine the relative positional relationships of each image. We hypothesize that images with a higher number of matching feature points are in closer proximity. However, it is not always the case that the two images with the most matching feature points are necessarily adjacent. We need to seek a **globally optimal arrangement**.



We have reformulated the issue of determining the relative positioning of a set of images into a **closed-loop graph search algorithm problem**. Images that have a higher number of matching points are more likely to be adjacent to each other. Furthermore, the first and last images in the sorted sequence are positioned next to each other, as the photographs were taken surrounding the object.



In the course of our experimentation, we have combined the **MST (Minimum Spanning Tree)** and **DFS (Depth-First Search)** methods to find the optimal solution for our problem. The MST is a tree structure that connects all vertices in a graph at the minimum total edge weights without forming a cycle. A commonly used algorithm for constructing MSTs is Kruskal's algorithm. By employing MST, we can identify the least costly way to connect all nodes within a graph. DFS is a graph traversal algorithm that uses recursion or stacks to explore along a branch of the graph until it reaches the last node, then backtracks to other branch extremities until all nodes have been visited. DFS can be utilized to search for possible paths or to check for the presence of cycles within the graph. **In summary, MST is employed to find the shortest path, while DFS is utilized to impose closed-loop constraints on the shortest path.**

|  object  |  CV   |
|---- | ---- | 
|  glass_cup  |   0.601   | 
|  glass_cylinder  |  0.783    |


|  object  | LB   | PB   |
|---- | ---- | ---- |
|  +MST&DFS |     +0.03   |   +0.02   |


The following is our code


