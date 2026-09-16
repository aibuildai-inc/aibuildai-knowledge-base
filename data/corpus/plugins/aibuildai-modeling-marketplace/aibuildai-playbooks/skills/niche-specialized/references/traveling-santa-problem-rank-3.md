# Solution of the prize winner (Rudolph)

Competition: traveling-santa-problem
Rank: #3
Source: https://www.kaggle.com/c/traveling-santa-problem/discussion/3763

<p>Here is a description of my algorithm for the Kaggle travelling santa competition :</p>
<p>The algorithm is divided into three phases :</p>
<p><strong>Phase 1 - Initialisation</strong></p>
<p>Essentially produce two tours for the problem by using the linkern program in the concorde library (<a href="http://www.tsp.gatech.edu/concorde.html">http://www.tsp.gatech.edu/concorde.html</a>). At this stages the two tours have a lot of common edges. A
 list of potential edges is constructed such that every edges in the initial two tours are present and also, for every nodes, the edges connected to its numneighbors closest neighbors.</p>
<p><strong>Phase 2 - Reparation</strong></p>
<p>The objective of this phase is to get a feasible solution. This is done by iteratively getting new tours by using the linkern program while increasing a penalty on conflicting edges. At some point (by default when there are less than 100 conflicts), to hasten
 convergence, no new conflicts are authorized. The algorithm is essentially this.</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<hr>
<p style="padding-left:60px"><em><span style="line-height:1.4em">tour1 = initial_tour1</span></em><br>
<em>tour2 = initial_tour2</em><br>
<em>penalty[e] = 1.0 for every edge e in initial_edgelist</em><br>
<em>num_conflicts = compute_conflicts();</em><br>
<em>while (num_conflict &gt; 0)</em></p>
<p style="padding-left:90px"><em>for every edges e in conflict</em><br>
<em>penalty[e] = penalty[e] * conflict_increase</em><br>
<em>edgelist = initial_edgelist</em><br>
<em>if num_conflict &lt; lim_conflict</em></p>
<p style="padding-left:120px"><em>removed from edgelist every edge that is in tour2 but not in tour1</em></p>
<p style="padding-left:90px"><em>solve tour1 with linkern considering edge in edgelist with cost[e] = length[e]*penalty[e];</em><br>
<em>edgelist = initial_edgelist</em><br>
<em>if num_conflict &lt; lim_conflict</em></p>
<p style="padding-left:120px"><em>removed from edgelist every edge that is in tour1 but not in tour2</em></p>
<p style="padding-left:90px"><em>solve tour2 with linkern considering edge in edgelist with cost[e] = length[e]*penalty[e];</em><br>
<em>num_conflicts = compute_conflicts();</em></p>
<p>&nbsp;</p>
<hr>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>At the end of the algorithm, with a good choice of parameters, the solution is typically around 6542000 but can be lower. Most of the time there is a gap between the length of the two tours&nbsp;for example something like 6542000 and 6535000.</p>
<p><strong>Phase 3 - Improvement</strong></p>
<p>In this phase we try to improve the solution generated in phase 2 by using a crude version of the Lin-Kernighan algorithm (Double-Lin-Kernighan) adapted to the two tours version and using the penalty used in phase 2. The idea, similar to what the Lin-Kernighan
 algorithm does, is to get a construction R1(1)-A1(1)-R2(1)-A2(1)-R2(2)-A3(2)-R4(2)-A4(2), where Re(t) remove edge e from tour t and Ae(t) add edge e to tour t. The adaptation to the two tours version lie in the fact that when a conflict is introduced by closing
 a tour in the substitution the algorithm can readily repaired it by switching to the other tour and removing the conflicting edge.</p>
<p>The substitution must have the following property:</p>
<p>1. It must form two tours (with possible conflicts)<br>
2. If the sequence is stopped before the end by a removed arc it could be closed (forming a tour) by adding an arc from the initial edge list.<br>
3. Given a metric, a sequence that start at the beginning and that end by a removed arc has always a positive value when evaluated according to the metric.</p>
<p>Two metrics are used in this phase. The first one (balanced/improved) aimed at closing the gap between the value of the two tours and decreasing both tours length while not augmenting the number of conflicts too much. The second one (feasibility) aimed at
 reducing the number of conflict while not augmenting the worst tour too much. The metric are calibrated such that when executed one after the other the two approaches produced a solution with slowly decreasing penalty and slowly decreasing worst tour value.
 The algorithm is:</p>
<p>&nbsp;</p>
<hr>
<p>&nbsp;</p>
<p style="padding-left:30px"><em>Make one pass of Double-Lin-Kernighan with metric 1 (balanced/improved) &nbsp; // get balanced solution with few conflicts</em><br>
<em>for i &lt; num_iter</em></p>
<p style="padding-left:60px"><em>Make one pass of Double-Lin-Kernighan with metric 2 (feasibility)</em><br>
<em>Make one pass of Double-Lin-Kernighan with metric 1 (balanced/improved)</em><br>
<em>edgelist = initial_edgelist</em><br>
<em>removed from edgelist every edge in tour2 but not in tour1</em><br>
<em>solve tour1 with linkern considering edge in edgelist with cost[e] = dist[e]*penalty[e];</em><br>
<em>edgelist = initial_edgelist</em><br>
<em>removed from edgelist every edge in tour2 but not in tour1</em><br>
<em>solve tour1 with linkern considering edge in edgelist with cost[e] = dist[e]*penalty[e];</em></p>
<p style="padding-left:30px"><em>Do phase2 to repair the few conflicts that remains</em></p>
<p>&nbsp;</p>
<hr>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>The third phase is quite good at reducing the initial gap between the tours. However it is not very efficient at further reducing the worst tour value and, passed a certain point, stopped being efficient at all (probably when the penalties are too high).</p>
<p>The final solution is obtained by removing the longest arc from each tours.</p>
<p><strong>Source code</strong></p>
<p>It can be compiled by the command <em>make all</em>. To run you need the linkern executable from the concorde distribution (<a href="http://www.tsp.gatech.edu/concorde.html">http://www.tsp.gatech.edu/concorde.html</a>). It should compile on most unix-like
 platform. I've tested it on ubuntu linux and macos X terminal.</p>
<p>tspsanta.cpp is the main program. It parsed the command line and called the whole algorithm or any phase individually.</p>
<p>PenLinkern.cpp and .h is the class that implement Phase I and Phase 2.</p>
<p>DblLinkern.cpp and .h is the class that implement Phase 3.</p>
<p>My best score was obtained by running the following command line (note that the parameter used are the default one and the score could vary since there is randomness the concorde linkern) :</p>
<p style="padding-left:30px"><em>santatsp doall santa_cities.csv tour1.txt tour2.txt edges.txt penalty.txt solution.csv 14 30 120 30 1.005 100 50</em></p>
<p>The command line parameters are:</p>
<p><strong>phase</strong> : (init, create, solve, improve, solution, doall)<br>
<strong>probname</strong> : the filename of the problem (default: santa_cities.csv)<br>
<strong>tour1filename</strong> : the filename of the tour1 (default: tour1.txt)<br>
<strong>tour2filename</strong> : the filename of the tour2 (default: tour2.txt)<br>
<strong>edgefilename</strong> : the filename of the edgelist passed to linkern (default: edges.txt)<br>
<strong>penaltyfilename</strong> : the filename of the penalty file (default: penalty.txt)<br>
<strong>solfilename</strong> : the solution filename (solution.csv)<br>
<strong>numneighbors</strong> : the number of closest neighbors considered in the init phase<br>
<strong>initduration</strong> : the duration of the linkern optimisation during phase 1<br>
<strong>solveduration</strong> : the duration of the linkern optimisation during phase 2<br>
<strong>improveduration</strong> : the duration of the linkern optimisation during phase 3<br>
<strong>penaltyincrease</strong> : the increase in penalty at every iteration an edge is in conflict<br>
<strong>conflictlimit</strong> : if the number of conflict is lower than this value, no new conflicts are permitted (default: 100).<br>
<strong>improveiter</strong> : The number of improvement iteration (default : 50).</p>
<p>To run a quick try (well not too long) I suggest :</p>
<p>solveduration : 30<br>
penaltyincrease : 1.025<br>
conflictlimit : 500<br>
improveiter : 5</p>
<p><span style="line-height:1.4em">Thanks to the Kaggle team for organizing this competition.</span></p>
<p>&nbsp;</p>
