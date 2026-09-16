# My optimal solution

Competition: santa-workshop-tour-2019
Rank: #42
Source: https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126374

I used Gurobi 9.0 with a non limited evalution license. Thanks Gurobi for giving me an avalution license. SCIP in ortools can only get me to 68910.94. 
- I fixed the occupancy of days at this array and run MIP model around +/-5 values of each day. 
`estimate_occupancies = np.array([300, 287, 300, 300, 286, 262, 250, 250, 271, 296, 300, 300, 279,
        264, 256, 273, 294, 282, 259, 228, 196, 164, 125, 300, 300, 295,
        278, 263, 257, 253, 279, 276, 252, 219, 188, 156, 125, 283, 271,
        248, 216, 184, 159, 125, 300, 282, 257, 226, 194, 161, 125, 286,
        264, 236, 201, 168, 137, 125, 266, 241, 207, 166, 125, 125, 125,
        253, 225, 190, 147, 125, 125, 125, 227, 207, 175, 129, 125, 125,
        125, 235, 220, 189, 147, 125, 125, 125, 256, 234, 202, 161, 125,
        125, 125, 234, 214, 181, 136, 125, 125, 125])`
With this scope, the MIP model run under 10 minutes on my macbook.
- Where did I get that estimated occupancies? I run LP model on 5000x10 preference cost CONTINOUS variables (from 0 to 1) + 100x176x176 accounting cost BINARY variables just like the guide from https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/120764 and added these quad constraints, zs is mine 100x176x176 accounting cost variables:
```
for i in range(0, N_DAYS, 2):   ## only need this for every two days
    for u in range(0, 176, 2):   # every two occupancies
        for v in range(0, 176):   
            two_vars = zs[i][u, v] + zs[i][u+1, v]   
            m.addConstr(two_vars*two_vars==two_vars)
```
These quad constraints are need for LP only, in MIP model we don't need it. In the LP model, they force the relaxation to use two adjacent occupancies for a day. This way, the LP model is not optimistic about the accounting cost in the relaxation. You don't need to run the LP model to the end, just stop it when a 67xxx solution has been found, about 25 minutes on my macbook.
- The solution sounds easy and fast, but it took me months to find out. Congratulation and thank you everyone, I can't get this score without refer to discussions in this great competition.
