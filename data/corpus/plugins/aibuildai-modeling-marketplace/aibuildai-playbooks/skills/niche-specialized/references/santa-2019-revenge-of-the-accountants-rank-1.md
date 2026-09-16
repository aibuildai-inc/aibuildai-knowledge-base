# Summary: how we found the optimal (?) solution

Competition: santa-2019-revenge-of-the-accountants
Rank: #1
Source: https://www.kaggle.com/c/santa-2019-revenge-of-the-accountants/discussion/126380

This is a (short?) description of what we did to find the best known solution for this challenge. Well, since this and the original challenge are pretty similar, we actually describe both.

Our basic solution approach is very similar to the original Santa challenge and the same that many teams used. We linearized the objective function and solved the resulting MIP using Gurobi (actually the incredible Christmas elves at Gurobi released the latest version just before the challenge started, this cannot be a coincidence ... ;)). For completeness we show the full model.

First, the assignment of each family to each day is pretty straightforward. For each family \\(f \in F\\) and each day \\(d \in D\\) we introduce a binary variable

  \\[x\_{f,d} \in \\{0,1\\}, f \in F, d \in D\\]

representing the assignment of family \\(f \\) to day \\(d\\). Each family must be assigned exactly once

  \\[\sum\_{d \in D} x\_{f,d} = 1, f \in F\\]

The matching objective simply depends on the day each family is assigned to:

  \\[\text{matching: } \sum\_{f \in F} \sum\_{d \in D} x\_{f,d} \cdot c\_{f,d} \\]

where \\(c\_{f,d}\\), \\(f \in F\\), \\(d \in D\\), denotes the assignment cost for family \\(f \in F\\) to day \\(d \in D\\).

Then we need the number of persons assigned to each day. Let \\(p\_f\\), \\(f \in F\\), denote the number of persons of family \\(f \in F\\), then the (integer) variable

  \\[n\_d \in \\{125, \ldots, 300\\}, d \in D\\]

denotes the number of persons assigned to day \\(d \in D\\). The correct number is found using the following constraints:

  \\[n\_d = \sum\_{f \in F} p\_f \cdot x\_{f,d}, d \in D\\]

This was the easy part. Now we need to model the non-linear penalty. The basic observation is that each term of the penalty objective depends only on two values: \\(N\_d\\) and \\(N\_{d+i}\\) for \\(i=1, \ldots, 5\\) (the original Santa challenge had only the terms for \\(i=1\\)) and each \\(N\_d\\) can only take the values \\(V := \\{125, \ldots, 300\\}\\). As many had realized, this can be linearized as follows:

First we introduce binary variables \\(n\_{d,v}\\), \\(d \in D\\), \\(v \in V\\) with \\(n\_{d,v} = 1\\) if and only if \\(n\_d = v\\) (if exactly \\(v\\) persons have been assigned to day \\(d\\)). Because \\(n\_d\\) has exactly one value we add the constraints

  \\[\sum\_{v \in V} n\_{d,v} = 1, d \in D.\\]

The \\(n\_{d,v}\\) and the \\(n\_d\\) variables are then connected using

  \\[n\_d = \sum\_{v \in V} v \cdot n\_{d,v}, d \in D\\]

As said before, each term of the penalty objective depends only on the value of two variables \\(n\_d\\) and \\(n\_{d+i}\\) or, equivalently, on \\(n\_{d,v}\\) and \\(n\_{d+i,v'}\\). Let \\(c\_{m,m',i}\\) be the value of one penalty term if \\(n\_d = m\\) and \\(n\_{d+i} = m'\\) for some \\(d \in D\\). Then the penalty objective is

  \\[\sum\_{d \in D} \sum\_{i=1}^5 \sum\_{m \in V} \sum\_{m' \in V} n\_{d,m} \cdot n\_{d+i,m'} \cdot c\_{m,m',i}\\]

(In the original problem only the terms for i=1 appeared). Of course, this is not a linear objective yet (because of \\(n\_{d,m} \cdot n\_{d+i,m'}\\). The standard approach is to introduce a new binary variable representing each product

  \\[y\_{d,i,m,m'} = n\_{d,m} \cdot n\_{d+i,m'}, d \in D, i=1, \ldots, 5, m,m' \in V\\]

This is still a non-linear constraint, but because exactly one of the \\(n\_{d,\cdot}\\) and one of the \\(n\_{d+i,\cdot}\\) is one, these constraints can be replaced by

  \\[n\_{d,m} = \sum\_{m' \in V} y\_{d,i,m,m'}, d \in D, m \in V, i=1, \ldots, 5\\]

and

  \\[n\_{d+i,m'} = \sum\_{m \in V} y\_{d,i,m,m'}, d \in D, m' \in V, i=1, \ldots, 5\\]

These constraints say that \\(n\_{d,m}\\) is one if and only if one of the "product-variables" \\(y\_{d,i,m,m'}\\) is one (and similar for \\(n\_{d+i,m'}\\)). The "true" penalty objective is now

  \\[\text{penalty: } \sum\_{d \in D} \sum\_{i=1}^5 \sum\_{m \in V} \sum\_{m' \in V} y\_{d,i,m,m'} \cdot c\_{m,m',i}\\]

That's basically all. For \\(i\\) restricted to 1 the model solves the original problem in reasonable time. There are a few tricks to speed up the computation, but they seem to be required only for the revenge challenge (see below).

The main difficulty for the second challenge was that the model gets much bigger (because of \\(i=1, \ldots, 5\\) and not only \\(i=1\\) we need much more variables for the linearized objective). Furthermore, IP solvers work by solving the LP relaxation of the model first and the larger model seems to be pretty challenging. In fact, Gurobi (and also Cplex) have big numerical trouble with solving the LP relaxation: the barrier algorithm stops with numerical problems, the simplex method takes forever.

This seems to make the problem more difficult. However, looking at the changed objective this is not true, at least from a practical perspective: it seems to be much easier to find good solutions for the second challenge (finding an optimal one and proving that it is optimal is hard). As far as I can tell, the reason why the second challenge seems to be easier is as follows. In the original challenge the penalty objective had a "combinatorial" property. Typically you do not want to have "big jumps" in the number of persons assigned to successive days because a big difference between \\(N\_d\\) and \\(N\_{d+1}\\) causes large penalty costs. However, if \\(N\_d\\) and \\(N\_{d+1}\\) are close, then the penalty costs are very small. (This motivates a simple heuristic: forget about the penalty objective in the MIP and add constraints \\(-30 \le N\_d - N\_{d+1} \le 30\\) for all d -- this gives you a quite good solution for the original challenge). There is one exception to this "rule": if \\(N\_d = 125\\) for some \\(d \in D\\), then the penalty objective for the term for \\(N\_d\\) and \\(N\_{d+1}\\) is zero, no matter what the value of \\(N\_{d+1}\\) is. In fact, if you look at the optimal solution of the original problem you will see that the number of persons assigned to successive days typically decreases from 300 to 125 over a few days in relatively small steps and then immediately jumps to 300. In order to find a good (or optimal) solution it is therefore crucial to identify the days which have exactly 125 persons (indeed, these days can be identified quite accurately by just looking at the number of persons that prefer a certain day).

The difference in the new challenge is that because not only 2 days but 6 days in a row are considered, having a single day down to 125 does *not* make the penalty cost to vanish. In fact, even if \\(N\_d = 125\\) and \\(N\_{d+1} = 300\\) (which would cause zero penalty in the original problem), each possible value for \\(N\_{d-1}\\) except \\(N\_{d-1} = 125\\) will cause large penalty costs for either \\(N\_{d-1}\\) and \\(N\_d\\) or \\(N\_{d-1}\\) and \\(N\_{d+1}\\). One would need to have several days with 125 persons in a row in order to get the zero penalty, but given the large number of persons (there are even more than in the original challenge) this is very unlikely. In other words, the "combinatorial" property of the original problem has been wiped out by the new objective.

Knowing this, how do we find a good solution? The key idea is to reduce the model size (by fixing many variables to 0). We did the following:

1. find a reasonable solution by ignoring the penalty objective and restricting \\(|N\_d - N\_{d+1}|\\) as illustrated above. Because the penalty costs are larger we forced the \\(N\_d\\) to be even closer (we used a difference of 10 if I remember correctly).

2. Given a solution with assigned number of persons \\(N\_d^*\\) to each day \\(d \in D\\), look for a solution where the number of persons for each day does not deviate too much from the given solution. In particular, we added the constraints

   \\[n\_{d,v} = 0 \text{ for all } d \in D, v \in V \text{ with } |v - N\_d^*| &gt; 20\\]

   and

   \\[N\_d^* - 20 \le n\_d \le N\_d^* + 20, d \in D\\]

Note that you do not have to actually remove variables or so, the preprocessor of your solver will eliminate them for you.

The size of this model is much smaller and the LP relaxation becomes tractable. We solved the model until we found a better (integral) solution. Once a better solution had been found we repeated step 2 with the newly found solution as starting solution. Eventually we found the best solution that is known to us and that we submitted. Of course, this is *no* proof that our solution is optimal, but we know at least that it is a very deep local optimum (although we are pretty sure that you can come up with a "hand-written" branch &amp; bound that scans the whole solution space and solves only restricted models that are actually tractable -- however this seemed not worth the effort ;))

There are many more tricks that can be applied to reduce the model size. In fact, we used the very same approach for the original challenge, too, because we were too impatient to wait for our solver for more than 20 minutes or so, and once you identify the "125-days" correctly, the approach that we used in the second challenge can also be applied to the first challenge to find the optimal solution pretty quickly.

That's all from us. Thanks for the nice challenge. We really enjoyed working on it during the holidays.
