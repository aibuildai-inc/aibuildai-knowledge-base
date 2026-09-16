# John does California Odyssey (with code)

Competition: santa-workshop-tour-2019
Rank: #13
Source: https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126254

I was very lucky to team with Alain and Stéphane.  Without them I would not have entered this competition as I was already engaged in another one.  And without them I would not have found the optimum that fast, therefore would not get a gold medal for sure.  Contributions from team members were all significant.  We came to this via different point of views, which was very fruitful.  

We started from this notebook https://www.kaggle.com/vipito/santa-ip/

It has several interesting components:
- A LP model for family to days assignment.  
- Max constraints on difference between successive days occupancy
- A local search to improve solutions

We then explored a number of variations and improvements.  When one of us was finding a solution it was shared with the other ones so that they can start from it in their next run.  We iterated over many models and runs.  Typically we would not let something run for more than a day.   Indeed, the better the starting point the better the end point!  And also, when using MIP models, the better the starting point the smaller the model as many variables can be set to 0 upfront.

The things we tried include: data/solution analysis , local search, LNS, linearization, approximation, simplification, symmetry breaking.  Let's look at each of those, in nor particular order.

**Local Search**

We started form the stochastic search of the public kernel, but then moved to a search similar to what is used in max flow algorithm: find a chain of family reassignment that keeps occupancy mostly unchanged and improve cost.  For instance, comparing two solutions found early in our endeavor we saw that they difference in only few places:

    Family: 261 83 - 67
    Family: 779 67 - 7
    Family: 798 35 - 45
    Family: 2926 1 - 35
    Family: 3215 25 - 83
    Family: 4716 45 - 1
    Same: 4994

If we look carefully, we see that moves can be chained:

    35-45-1-35
    25-83-67-7

We have one 3 cycle and one 3 path that capture all changes.

We coded a systematic search for chains up to a given length.  This was way more effective that a brute fore on possible family swaps.

**data/solution analysis**

After our first few solutions we found that the distribution of choices ranks was highly skewed.  Most families had one of their first 4 choices.  A first consequence is to limit model complexity by only considering choices up to 4, or 6, depending on the runs.  Unless mistaken, all families got one of their top 6 choices in our optimal solution.  We relaxed this at the end when we proved optimality.  

Another example of data analysis was to look at `gap(d)` which is absolute differences of occupancy of a day d. n Here is a plot gap(d) as function of the occupancy of the day for a solution of cost 69158.xxx



We see there is a simplex convex hull, which can suggest additional conditional constraints.  For instance:
    if (number(d) &gt;= 126),  then (number(d)-number(d-1)) &lt;= a-b*number(d) 

where `number(d)` is the occupancy on day `d`, `a`and `b`two parameters we set for each run.

**Cost approximation**

The accounting cost function is non convex, which makes it tricky to optimize.  Here is a log plot of it capped by a high value (100,000 I think)..

.png?generation=1579180264664895&amp;alt=media)

Even if non convex, it has some clear properties.  It is increasing with the gap more than linearly.  This led to the idea of minimizing the sum of their squares.  Unfortunately this was not very effective.

Other approximations were based on conditional constraints of the form:

    if number(d) &gt;= a, then cost(d) &gt;= piecwise(gap(d))

i.e. approximating the cost by piecewise functions that minor the actual cost.

When we approached the end of our odyssey we switched to an exact representation of the cost via the now well known 3M variable model, first shared by @hengck23 .  We actually tried this model the first day we entered the solution, but solving it from scratch did not seem feasible at the time.

**Large Neighborhood Search**

Local search is powerful, but it does not exploit the flow structure present in the model and it does not allow for massive changes in family assignments.  We used another form of search that could lead to large changes in family assignment.  The idea was to start from a solution, keep its occupancy per day, then constraint occupancy to be close to that initial occupancy, and solve the problem as a MIP.   This is very effective to explore a large neighborhood of the initial solution, hence its name.  It led us to find the deep local optim at 68910.94.  But this could not lead us to an optimal solution.  Reason for that is clear when we look at occupancy per day for our optimal solution and for  that deep loacl optimum:

.png?generation=1579181603445237&amp;alt=media)


We see that the optimal solution has one extra dip to 125 compared to the other solution.  There is no way our local search or our large neighborhood search would have found it.

**Exact model**

After few days we switched to an exact model (see description below) and used various subset of it.  Subsets can mean: limiting family choices to top K (k = 4, or 6, in practice).  Limiting the max value of the gap.  These limits were implemented by setting variables to 0 before solving the problem.  The model was always initialized by a solution (mi start).  Same for variables used to represent the cost.  Those with a large coefficient were set to 0.

Then all the tricks above were used to get new solutions quickly from known solutions.  Another trick was to fix some day occupancy to 125.  At a point, starting from a solution of cost 68914.2801, limiting to best 6 choices, and fixing 4 day occupancy to 125 we found an optimal solution overnight.

**Optimality Proof**

Finding a solution of optimal cost is not the same as proving there is no better solution.  We had a slight hope that none of the teams in front of us on the LB found the actual optimum.  Our run showed they did find the optimum.  The model is very similar to the 3M model shared publicly.

A binary variable x for each pair (family,day)
A binary variable y for each pair (day,occupancy of the day)
A binary variable z for each triple (day, occupancy of the day, occupancy of next day)

The constraints are straightforward except one that was disclosed by @hengck23 : summing the variables z along one of the last two axis is equal to one of the variable y.  
    
Code for a cplex implementation of the full model is available at https://github.com/jfpuget/Kaggle_Santa_2019.  We ran this on a machine with 20  cores at 2.6GHz .  It uses 16 GB.  It proves optimality in less than 3 hours, when mip start is our optimal solution.

Before that run we tried to add symmetry breaking constraints with the hope of speeding proof.  Indeed, once families start to not getting their preferred choices a lot of family assignments yield the same cost.  Symmetry breaking as effective in a way as it halved the number of nodes for the proof, but running time was a bit larger.

Edit: Our full model is exactly the same as the one described by @frankfisk : https://www.kaggle.com/c/santa-2019-revenge-of-the-accountants/discussion/126380
