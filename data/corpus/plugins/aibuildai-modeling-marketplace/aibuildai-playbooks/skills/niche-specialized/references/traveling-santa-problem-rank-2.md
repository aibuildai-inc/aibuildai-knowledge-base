# Source code of the prize winners

Competition: traveling-santa-problem
Rank: #2
Source: https://www.kaggle.com/c/traveling-santa-problem/discussion/3652

<p>Finishing in 2nd place, I was required to publish my source code. It is kind of embarrassing compared to the detailed and amazing explanation provided by the winner team, but I will try to share the main points of my approach below.</p>
<p>I use only simple code written by me (a single Java file attached, with about 1,400 lines). This may be a disadvantage is this kind of competition, but a major part of the fun for me is building everything from the scratch, although usually suboptimal :-)</p>
<p>The main methods of my solution are:</p>
<p style="padding-left:30px"><strong style="font-size:13px; line-height:1.4em">1) buildNearestNeighbor()</strong><span style="font-size:13px; line-height:1.4em">: build an initial solution using nearest neighbor. At the end of this step and during all the other
 steps the solution must remain valid, i.e., connect all points without using a connection between a pair of points more than once.</span></p>
<p style="padding-left:30px"><strong style="font-size:13px; line-height:1.4em">2) moveSegments():</strong><span style="font-size:13px; line-height:1.4em"> remove a segment (with one or more consecutives edges) and try to fit it somewhere else.</span></p>
<p style="padding-left:30px"><strong style="font-size:13px; line-height:1.4em">3) breakAndReconect()</strong><span style="font-size:13px; line-height:1.4em">: remove few edges from the same path and try to reconnect them in a different order (checking all permutations).</span></p>
<p style="padding-left:30px"><strong style="font-size:13px; line-height:1.4em">4) moveEnds():</strong><span style="font-size:13px; line-height:1.4em"> try to move the endpoints.</span></p>
<p style="padding-left:30px"><strong style="font-size:13px; line-height:1.4em">5) fixRegion():</strong><span style="font-size:13px; line-height:1.4em"> similar to breakAndReconnect, but working with the two paths at the same time. It uses a lot of pruning to
 avoid useless permutations.</span></p>
<p><span style="font-size:13px; line-height:1.4em">Other observations:</span></p>
<ul>
<li><span style="font-size:13px; line-height:1.4em">The “movements” methods (2, 3 and 5) can all be tuned, changing the number of neighbors to consider, running time, enable/disable Simulated Annealing (SA) and its temperature, etc. It seems that SA helped
 a lot.</span> </li></ul>
<ul>
<li><span style="font-size:13px; line-height:1.4em">Although methods 2 and 3 only work with one path at a time, they check, when there a movement that will improve the solution that can’t be made because of a single connection that is already in use by the
 other path, if there is a simple movement that “releases” this connection with a cost smaller than the gain of the first movement.</span>
</li></ul>
<ul>
<li><span style="font-size:13px; line-height:1.4em">From time to time, if improvements have been found, it saves the new solution to a file, which can be loaded in the next execution.</span>
</li></ul>
<ul>
<li><span style="font-size:13px; line-height:1.4em">Usually I ran 3 or 4 instances with different parameters during the night and took the best one on the next day to continue from there.</span>
</li></ul>
<ul>
<li><span style="font-size:13px; line-height:1.4em">There is an auxiliary method that I called “equalize()” which swaps a pair of edges between the paths, making their length similar. For example: IF path 0 is :
<strong>... - A - B - ... - C - D - ...</strong> and path 1 is .<strong>.. - A - C - ... - B - D - ...</strong> AND len(path 0) &gt; (len path 1) AND (dist(A,B) &#43; dist(C,D)) &gt; (dist(A,C) &#43; dist(B,D)) THEN inverting everything between B-C in both paths is a valid
 movement that usually will make the length difference smaller.&nbsp;This helped a lot because before implementing it I couldn’t control the big gap that other random movements were generating. So the goal was simplified to optimize any path, not only the longest
 one, because it was easy to equalize their length. It also shuffles things a little bit, and that helped other methods finding improvements, so it was called from time to time.</span>
</li></ul>
<ul>
<li><span style="font-size:13px; line-height:1.4em">I kept a list of nearest neighbors for each point using a 4-quadrant division (it took the closest point of each quadrant and ordered them by distance, then the second closest and so on).</span>
</li></ul>
<ul>
<li><span style="font-size:13px; line-height:1.4em">Running the attached code, as it is (with the hard coded parameters), allowing 3 hours (one hour for each of the mains steps - 2, 3 and 5), gives a score around 6,580,000. The rest of the gain came from running
 for a longer time and running again and again with different parameters.</span> </li></ul>
