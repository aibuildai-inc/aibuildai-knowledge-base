# 2nd place solution

Competition: santa-2022
Rank: #2
Source: https://www.kaggle.com/c/santa-2022/discussion/379086

Hi everyone !

First I would first like to thank the Kaggle team for this great challenge! I think it was very well calibrated and lots of fun to work on. I really enjoyed the arm twist which added a nice layer of complexity to the problem. Kudos to the creator(s)! 

Congratulation to C-number and Kibuna for getting to the local(?) optimum first! I got there 48 hours late. I may possibly have found it earlier if I had started merging my solutions sooner but I did not expect at the time that anyone would hit an optimum so quickly. Well done Newtonians! Congratulation also to Rafbill for his impressive head-start and to all the other kagglers who made this competition fun and exciting!  😊


### Main steps

**1.** I first tried to acquire an intuitive grasp of the arm movements. It looks similar to the usual \\( L^2 \\) problem at first sight but it turns out to be much trickier to perform inverse kinematics because of the lack of rotational symmetry (here we only get the symmetry of the \\( \mathbb{Z}^2 \\) lattice). I found out that I like to think of the arms as some sort of nested [slidding puzzles](https://en.wikipedia.org/wiki/Sliding_puzzle). This way of looking at the arm movements really helped me figure out conditions required to make a tour "liftable".

**2.** It was clear from the start that the configuration space is too big to work on it directly. So, as I guess most competitor did, I chose to factorize the problem by
      **(i)** Finding a low cost tour on the image space (for the \\( \sqrt{L^1} + \hbox{color} \\) distance).
      **(ii)** Lifting this tour from the image space onto the configuration space with minimal additional cost. 
      
   At first, I thought that going from (i) to (ii) would incur some cost due to reconfiguration of the arms needed to get a valid solution but, ultimately, it turned out that it is more effective to directly add constraints on the image tour so that it can be lifted at no additional cost. 

**3.** By looking at the LB, I realized early on that the cost difference between a tour on the image space and configuration space should be very small. Indeed, just a few days after the competition started, there were scores on the LB which were only a few tens of points above the best TSP tours I could generated. So I began generating as many TSP tours on the image space as possible. I figured they may be useful later... During that time, I looked for a way to lift a tour from the image space to the configuration space.

**4.** It took me some time to find a effective method to convert a tour from the image space to the configuration space. I thought about using a MIP solver for that task but I am not familiar with these tools and I prefer to code. Instead, I wrote a custom program in C++. On my 4 years old Ryzen7 desktop computer, it usually takes under 15 minutes to convert a tour on the image space into a valid solution (provided that the tour is really liftable of course). This could certainly be improved but it worked well enough for my needs. More details about the program/algorithm in paragraph **A** below. 

**5.** Once I had a reliable way to convert an image tour to a solution, I started to look how to minimize the reconfiguration cost. Ultimately, I discovered that being "liftable" is a rather soft requirement. Most hard constraints are located around the origin i.e. at the beginning and end of the path because of the specified arm configuration at \\( (0,0) \\). In particular, the most important constraint concerns the duration of the first/last excursion of the path in the half space \\( x\geq 0 \\). By trials and errors, I came up with a set of necessary (and mostly sufficient) conditions for a path to be liftable. More details about that in paragraph **B**. 

**6.** I incorporated these conditions as a "penalty" in a custom TSP solver which is a crossover between LKH3 and a pure C++ TSP solver that I wrote from scratch for the "Santa Prime Maths 2018 challenge". Running this TSP solver while slowly increasing the penalty, I obtained "liftable" tours with scores around 74076.

**7.** The final push was obtained by using all the (mostly unconstrained) tours I had gathered in the first weeks of the challenge. I used merging with IPT and some limited genetic algorithm to reach the final score of 74075.706541... In fact once I started merging tours, the solution popped up very quickly: I think it took less than 30 minutes to go from 76 to 75.706541...


### A. Lifting of a tour from the image space to the configuration space. 

I will make the code of the "lifting" program available on [my github page](https://github.com/vindar/) soon for anyone interested (but beware that the code is very dirty, sorry...).

Given a tour on the image space, we can decompose the path in 5 parts \\( A,B,C,D,E \\) such that:

$$ (0,0) \overset{A}{\longrightarrow} \hbox{corner 1} \overset{B}{\longrightarrow} \hbox{corner 2} \overset{C}{\longrightarrow} \hbox{corner 3} \overset{D}{\longrightarrow} \hbox{corner 4} \overset{E}{\longrightarrow} (0,0) $$

Noticing that the corner pixels have a unique arm configuration associated with them, it follows that the lifting of the path may be performed independently on each sub-path \\( A,B,C,D,E \\) and there will be no problem in joining them together (i.e. no reconfiguration needed). Also, it turns out that paths \\( B,C,D \\) between corners are always straight lines so lifting then is trivial. Therefore, it only remains to lift path \\( A \\) onto the configuration space (and the same procedure can be used to lift \\( E \\) by simply considering the reverse path \\( \tilde{E} \\) from \\( (0,0) \\) to \\( \hbox{corner 4} \\) ).

The algorithm used to lift path \\( A : (0,0) \to \hbox{corner 1} \\) is a classical random search with a few additional tricks. Basically, having lifted the path onto the configuration space, starting from \\( (0,0) \\) and  up to some given pixel, the algorithm tries to find an adjacent configuration located at the next pixel on the path. If there are several possibilities, it chooses one of them at random. Otherwise, the algorithm backtracks a random number of steps with geometric distribution with mean given by a temperature parameter T (just like in simulated annealing). However, this simple algorithm does not perform very well because the search space is too big and the lifted path must sometimes perform very unlikely moves... I added several improvements to the algorithm above but the three tricks which really made a difference are:

**1.  Making the temperature fluctuate**. When T is large the algorithm backtracks deeper so it helps to prevent getting stuck in narrow passages whereas for T small it explores the local neighborhood more thoroughly. 

**2. Adding a fluctuating random drift for each arm in a given direction**. This drift is used to bias the choice of the arm to move when more than one arm is admissible. Doing this biasing is especially important for the largest links (of sizes 64 and 32). Indeed, the largest link sometimes needs to move at lot during a short period. However, if we choose the moving arm uniformly at random, then the probability to select the largest link often enough becomes very small (of order \\( e^{-c n} \\) where \\( n \\) is the number of moves). On the other hand, whenever the drift aligns with the number of moved required, then this same event become a typical gaussian fluctuation (with probability of order \\( 1/\sqrt{n} \\) which is much higher). This strategy is, in a way, similar to [the proof of Cramer's large deviation theorem](https://djalil.chafai.net/blog/2018/03/09/tutorial-on-large-deviation-principles/) where exponentially small events are made "typical" by tilting the underlying probability distribution.

**3. "Un-knitting" the path**. When we encounter a dead-end, we can compute the distance (i.e. reconfiguration cost) needed to go to the next pixel. Then, going backward, we can try to recursively perform local change to this configuration path in ways that decrease the reconfiguration cost at the end of the path (for example, it is sometimes possible to swap the arm that moves at a given step without affecting the rest of the configuration path). This "un-knitting" procedure is, I believe,   closer to how an MIP solver might proceed to lift a path...


### Conditions for a tour to be "liftable": the penalty function. 

As I mentioned above, I like to think of the arm movement as sliding puzzles. Since a picture is better than a thousand words, in the case of a \\(17 \times 17 \\) picture with 4 links \\( (1,1,2,4) \\): 



Here, there are 4 nested blue squares of decreasing side lengths \\( 9, 5, 3, 1 \\), each one centered at the position of partial arm sums and the smallest unit blue square pinpoints the final arm position. With this representation, the arm movement rule translate to: *"each blue square can slide inside its larger blue square but it must always stay in contact with its boundary (and the largest blue square must stay in contact with the image boundary)"*. 

With this representation, the starting configuration at \\((0,0) \\) looks like: 



Looking at this picture, it is clear that a valid configuration path can only enter the half-space \\(x < 0\\) after moving the largest blue square at least 64 times vertically (in the up direction if the path enters \\(x < 0\\) at some position \\( (x_0, y_0) \\) with \\( y_0 > 0 \\) and in the down direction otherwise). By reversal, a similar condition is also required at the end of the path. I found that these conditions for the first and last excursions are the most important ones in order to get a "liftable" tour but they are not sufficient by themselves. I considered three more conditions:

**1. Subsequent excursions**. After the end of the first excursion i.e. when the path enters \\( x < 0  \\) at some point \\( (x_0,y_0) \\),  we can again split the space, but this time horizontally, and look at the first (later) time when the path exits \\( y < 0 \\) (if \\( y_0 >0 \\) ) or \\( y > 0 \\) (if \\( y_0 < 0\\) ). With the same reasoning as before, we can deduce a lower bound on the number of horizontal moves that the largest link must perform during this excursion. By induction, we can repeat the argument for all the subsequent excursions of the path (alternatively splitting the space vertically and horizontally)...

**2. Do not move to early to the right**. The starting position at \\( (0,0) \\) depicted above also clearly shows that the path cannot go "too fast" to the right at the very beginning without first moving a little bit up and down in order to "unlock some of the blue squares". In particular, the first move must be vertical. This condition is only restrictive for the very first steps (say the 15 first steps). Of course, the same condition also holds at the end of the path by symmetry.

**3. Breaking the lines at distance 1 from the image border**. This last condition is a bit surprising: a liftable path cannot travel all the way along both a vertical and an horizontal lines which are exactly 1 pixel away from the image boundary (i.e. the lines with equation \\( x = \pm 127 \\) and \\( y = \pm 127\\) ) otherwise a reconfiguration with cost \\( \sqrt{2} - 1 \\) would be required. 

I do not think that the set of conditions above is really sufficient to insure that a tour is liftable but, in my tests, most tours fulfilling them were indeed liftable... However, condition **1.** concerning  the "subsequent excursions" of the path is very costly to compute (because we must inspect the whole path) so I only enforced it at the end of a search but not during local search for k-opt moves... 


## Final remarks: computing TSP score with high precision

I suspect other competitors encountered the same problem: in order to use LKH (or another TSP solver) one must to convert the floating point weights to integer values, for example, by multiplying each values by 10000 and then rounding them the closest integer. However, a significant precision is lost in doing so. My first solution to overcome this difficulty was to update my TSP solver to use 64 bits values instead of 32 bits so I could scale the values by \\( 10^{10} \\)... 

However, there is a better solution that make use of the fact that the original PNG image was encoded in 24 bit RGB colors (8 bits per color channel). And indeed, if we look at the .csv file containing the colors, we can check that the floating points numbers inside only take the \\( 256 \\) different values \\( \frac{i}{255.0} \\) for \\( i=0..255\\). Therefore, in the TSP solver, we can instead set the color weights to be of the form \\(k i\\) for \\(k \\) large and then rescale the values \\( \sqrt{j} \\) appearing in the displacement cost accordingly. This improves precision significantly. However, when doing this, we are implicitly assuming that all the increments \\( \frac{i+1}{255.0} - \frac{i}{255.0} \\) are all equal which, although mathematically trivial, is unfortunately not true in the .csv file because we are reading values previously computed with 64 bits double precision... In fact these increments usually differ at the 17th digits. This means means that *two tours that would have the exact same score on the original PNG image will have different scores when computed using the values written in the csv file*. And  indeed, I have found hundreds of tours at the current optimum which all have different costs w.r.t the CSV file but  are all equivalent w.r.t. the PNG image.

Fortunately, because we are adding up 66000 numbers and because two tours which are equivalent on the PNG image usually differ by at most few hundred edges, the difference in scores does manifest itself before the 19th digits and therefore they all have the exact same score when rounded to double precision (which can only store up to 17 significant digits) as is done on the leaderboard. Because of this fact, I would bet that the last two digits of all 4 top scores are exactly equal (up to double precision) with value: 74075.706541690054. Could a kaggle organizer possibly confirm if this is the case ? And thanks again for this great challenge 😃
