# about my tourney to 68888.04

Competition: santa-workshop-tour-2019
Rank: #39
Source: https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126237

Thanks organizers for this competition!

Here is a way I approached optimum:
69400: iterative solution (about 5 iterations) based on cbc_solver and heuristics.

cbc\_solver optimized only preference cost (used 5000x5 variables milp formulation),
also daily occupancies were constrained by (prev\_iteration\_daily\_occupancies-band,prev\_iteration\_daily\_occupancies+band), where band=[6-8]

heuristics was simular to @xhlulu kernel but in c++ choosing subsets of size &lt;= 9, limiting family choices to choice\_0,choice\_1,choice\_2.

69197: reached with cplex solver within two hours (actually was limited by two hours of usage in watson studio cloud)
used milp formulation simular to @hengck23 (100x176x176), removed vars with acc cost&gt;6020 
and also vars which are out of (optimal\_lp\_solution\_daily\_occupancy - band, optimal\_lp\_solution\_daily\_occupancy + band) where band = 60
total about 1.5M variables,
seeded with 69400 solution

68910: reached with xpress solver in 8 hours with reduced milp formulation simular to 69197 (band=80)
seeded with 69197 solution, 

68888.04: reached with xpress solver in 8 hours with full milp formulation simular to @hengck23 (removed vars with acc cost&gt;6020)
seeded with 68910 solution, 

P.S. It was crucial to seed solver with good solution, reduce number of variables in formulation as much as possible and use a good commercial solver. I am not from academia, so could not get license for cplex or gurobi, lucky enough to get free 1 month full license for xpress on artelys.com:)
