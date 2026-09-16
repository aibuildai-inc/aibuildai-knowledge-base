# [21st place] Reflections on Santa 2021

Competition: santa-2021
Rank: #21
Source: https://www.kaggle.com/c/santa-2021/discussion/300901

This was a really interesting problem which was helpfully amenable to solution either through mathematical insights into its symmetry, or by the brute force of computer power, or even better by a combination of the two.

Like many Kaggle competitions, time spent reading the forum, working through the notebooks, and experimenting with one’s own adaptations of the code, was rewarded with nuggets of knowledge and understanding that helped unlock the higher level of this game. 

**Level 1:** 2507-2486

[The Santa 2021 TSP Baseline - [2500]](https://www.kaggle.com/cdeotte/santa-2021-tsp-baseline-2500) by @cdeotte

This was a really helpful starter to get into the problem, but beneficially for the competition its approach had a ceiling somewhere in the2480s. It is based on defining the three substrings using the order in which the permutations occur in the 5906 superpermutation, and then using the LKH TSP solver to minimise each substring separately. With this approach we have very limited flexibility to move permutations between substrings.

**Level 2:** 2484-2481

[MinMax CTSP](https://www.kaggle.com/kostyaatarik/minmax-ctsp)  by @kostyaatarik  

This notebook helped me make a key step forward, as it represented the problem as a Coloured TSP in LKH and added some code to change the objective function so that it now minimised the maximum length of any substring. This was evidently the correct minimisation to be performing, but the coding required to patch the objective would have been beyond my capabilities.

However, at this point I was still splitting into three sets of permutations based on the ordering of the 5906 superpermutation.  Both my own scores and a look at the division between sub-2440 and 2480+ scores on the LB at that point suggested that there was some ‘magic’ required to get from 2480 to 2440.

This seemed likely to involve the way in which permutations were assigned to substrings, and this somewhat cryptic [thread](https://www.kaggle.com/c/santa-2021/discussion/292998) in the Discussion further suggested that this might be related to the 5913 recursively generated superpermutation, probably a better candidate for symmetry-related properties than the shorter 5906, 5907 or 5908 ones. However, my own efforts had not managed to split up any of these superpermutations in the magic way.

**Level 3:** 2457-2439

The answer to this conundrum appeared in the notebooks

[Mathematic laws about super permutation](https://www.kaggle.com/w3579628328/mathematic-laws-about-super-permutation) by @3579628328

which suggested how to split the 5913 in a 3-fold symmetric way and

[Santa Baseline - 2481](https://www.kaggle.com/ks2019/santa-baseline-2481) by @ks2019

which did this in practice, giving substrings of length 2483 without wildcards (2481 with) that unlocked the potential of the MinMax CTSP. My very first minimisation starting there gave a then-surprisingly good (to me) score of 2457, and this method continued with roughly 7 nine-hour minimisations twice a day for about a week before getting stuck at non-wildcard lengths of 2441_2440_2440. Two subsequent repeats with slightly different minimisation protocols later stuck at 2442_2442_2441 and then 2441_2440_2440. It would be interesting to know if anyone managed to reach 2440_2440_2440 with the MinMax CTSP in LKH.

Up to this point, I was using

[Wildcard Postprocessing Using Dynamic Programming](https://www.kaggle.com/yosshi999/wildcard-postprocessing-using-dynamic-programming) by @yosshi999 

to add wildcards. This got the score down by 2, but left me stuck on 2439. It was time to team up.

**Level 4:** 2432-2428

The team-up with Rob @robikscube, Daniel @danieldias and Xuxu @xuxu1234 worked really well, in an extremely distributed manner. I will give only the briefest overview of my colleagues’ work.

Rob generated a 2440_2440_2440 non-wildcard solution. Daniel coded up the wildcard problem as an asymmetric TSP on Gurobi, using one substring at a time. Xuxu moved permutations around the substrings to optimise the benefits of wildcards, and placed our wildcards in the perfect places to reach 2428.

Congratulations to Daniel on becoming a Competitions Expert and to Xuxu on gaing the Competitions Master title.

Thanks to everyone in the community who analysed the problem in the discussion forum and posted helpful notebooks. Many of us benefitted from helpful hints and insights. For me, to finish in 21st place was a personal best in a medal-bearing competition, and I hope you all enjoyed the competition as much as I did.
