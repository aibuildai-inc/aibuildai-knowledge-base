# How to win Santa's Workshop Tour

Competition: santa-workshop-tour-2019
Rank: #1
Source: https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/127427

First, I would like to thank @inversion for preparing this nice competition. It was really fun.
Second, sorry that it took so long time till I wrote this post.
Third, yes most of you were right. I also used mixed integer programs (MIPs).

But now let's start!

My way to an optimal solution started with recognizing that the “nonlinear” objective function can be linearized by enumerating all possible combinations of people who may visit on one day and on the day after. 
After that, I had a short look into the data and believed that not many families will be assigned to a non preferred day. Nevertheless, I didn't want to remove this possibility completely. That's what led me to the following

# Mixed Integer Linear Programming Relaxation



* ``x_{f,d^f_p}`` is the binary variable which is ``1`` iff family ``f`` is assigned to its preference ``p`` for ``p=1,...,10`` and ``x_{f,d^f_{11}}`` is ``1`` iff the family is not assigned to one of its prefered days, where ``d^f_{11}`` is set to “day” ``101``.
* ``y_{d,i,j}`` is the binary variable which is ``1`` iff day ``d`` has ``i``  and day ``d+1`` has ``j`` people assigned. Note that we also introduced variables ``y_{100,i,j}`` for ``i``≠``j`` , for the ease of presentation. In this setting we can fix every variable ``y_{100,i,j}`` to ``0`` for ```i``≠``j``. Clearly, I didn't add these variables in my implementation.
* ``z_d`` is the “continuous” variable representing how many people are assigned to day ``d``.
*  Term ``(1)`` is the objective function which we want to minimize, where ``pc(p)`` represents the preference and ``ac(i,j)`` the accounting cost.
* Equation ``(2)`` ensures that each family is either assigned to one of its prefered days or to “day” ``101`` representing that the family is not assigned to one of its preferences.
* Equation ``(3)`` ensures that day ``d`` is assigned to a number of people visiting on this day and to a number of people visiting on day ``d+1``.
* Equation ``(4)`` is in some way “flow conservation” ensuring that the number of people of  consecutive days coincide.
* Equation ``(5)`` couples the consecutive day variables with the day quantity variables.
* Inequality ``(6)`` ensures that the number of people assigned to a day is at least the number of people assigned to that day which they prefer, where ``n_f`` is the number of family members of family ``f``.
* Equation ``(7)`` ensures that the number of people assigned to all days equals the number of family members. 

**Note, that in general a solution of this MIP don't have to be feasible for Santa's problem. Furthermore, I believe it can be as challenging as to solve Santa's problem from scratch to make a solution of this MIP feasible, if the data is bad .**

Nevertheless, the data did not look that bad and it turned out that I never had to “repair” a solution.

After roughly two hours Gurobi found a high quality solution and naturally I immediately submitted it to the leaderboard. Due to a mistake in my implementation this resulted in a solution which was scored with a value of &gt;``34145044298``, the overall worst score which was shown on the leaderboard in the whole duration of this competition.
After fixing the bug I got a solution with a value ≤``74589``. After three hours Gurobi produced a solution with a value of ≤``70913`` which did not further improve within 24 hours.
Nevertheless, I did not used this solutions, since I worked in parallel on a reduction on the number of variables.

# Lower Bounds, Upper Bounds, and Size Reduction

Since the number of variables is huge, I was interested in lower bounds on the preference costs (``LB_pref``) and upper bounds on the optimal value for the whole problem (``UB_opt``). The reason for this is, that given ``LB_pref`` and ``UB_opt`` we can bound the accounting costs from above by ``UB_opt - LB_pref``. This led me to the following MIP formulation, only optimizing the preference costs (8). This program was solved to optimality in less than a minute with ``LB_pref``≥``43622``.



**So, I had a lower bound on the preference cost: ``43622``**

For the upper bound we are lucky, since kaggle provides public leaderboards. At this time @wataorz was in top position with a solution score of ≤``70888``. Thus, the accounting costs are bounded from above by ``27266``.
To Further improve the lower bound on the preference costs I removed all variables ``y_{d,i,j}`` with larger accounting penalty than ``UB_opt - LB_pref``≤`` 27266``, added a constraint bounding the accounting costs (9), and solved the following programm.



**This program runs 10 minutes and gives a lower bound on the preference costs of ``LB_pref ``≥``54412``**

Note, that if we have an improved lower bound on the preference costs or an improved upper bound on the optimal costs we can rerun this program to get possibly an improved lower bound on the preference costs. In particular, I could have rerun the program directly but, since our lower bound improved, but I decided no to.

After that, I removed all variables ``y_{d,i,j}`` with penalty strictly larger than ``UB_opt - LB_pref``  from the first MIP and solved it.

**This produces a solution with value ≤``70134`` in roughly 70 minutes.**

Since I had now a good quality solution, I decided to try an improvement step.

# MIP Large Neighbourhood Search 

The last “tool” I used was a MIP representing a “large” neighbourhood search. Given a feasible solution, it restricts the number of people for each day to a given threshold ``TR`` from the number given by the initial solution (10). The day load a day ``d`` of the start solution is represented by ``l_d``. Again, I removed all variables ``y_{d,i,j}`` with penalty strictly larger than ``UB_opt - LB_pref`` and all variables not within the threshold.



I am not sure but I think I ran this program with threshold values ``20``≤``TR``≤``120`` which led me to an optimal solution. For one large ``TR`` I solved the program to optimality. So, I knew that in this huge neighbourhood there is not better solution. I had not much hope to solve the program to optimality with larger values. Thus, I decided to try to “prove” optimality and ran the first MIP with all bounds I had and my best solution.  It took about a day, but then Gurobi were proved optimality. For this run I changed the parameters of Gurobi to aggressively work on the bound.

Note, that my work was not that straightforward how I presented it here. I did many things in parallel and ran the above MIPs with new start solutions and improved bounds.

I hope you have fun with this post.

**Please let me know, if you see any mistakes or have questions!**
