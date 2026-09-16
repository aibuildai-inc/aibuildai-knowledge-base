# 3rd Place Solution: An Open Secret - Gather, Ridge, Repeat

Competition: playground-series-s4e9
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s4e9/discussion/537029

As mentioned in [my previous writeup](https://www.kaggle.com/competitions/playground-series-s4e8/discussion/531823), I was going to step back and participate more judiciously this month, and so I did. I'll go into the details shortly, but let me first explain the headline, which shares the open secret of how my approach was essentially the same as before:

<b> Gather: </b> Collect OOF predictions.
<b> Ridge: </b> Build an ensemble - you don't necessarily have to use Ridge Regression, but it generally does very well, and was a good fit for some wordplay 😀
<b> Repeat: </b> Literally, that - and also to complete the wordplay, of course.

Before going into details, I'd like to acknowledge the generosity of those who shared their insights, findings and code, including but not limited to

@alexryzhkov (and the AutoML team), @roberthatch, @omidbaghchehsaraei, @cdeotte, @siukeitin,  @ravaghi, @oscarm524, @ravi20076, @tilii7, @ricopue, @serhiikravtsov, @allegich, @backpaker 

<b> Phase 1 : Lurking </b>
I had a multi-week LLM project due on the 17th, and had to also review 3 projects by my peers in that course within a week after that. There was also an ongoing health emergency within the family. So it was quite clear that my participation would be quite limited until the last week or so, if not throughout. I kept an eye on the competition in this phase, checking out discussion posts and interesting public notebooks. I occasionally submitted a model to see how it scored, and gave into the temptation of some "blind blending" every now and then. It was great to have @cdeotte participating this month, as he made several insightful posts (inviting insightful responses by @siukeitin and others), and also generously shared notebooks.

Based on the competitions over the last few months (and especially last month), I figured that the day 1 solutions by Light AutoML Testers (@alexrhyzkov et al.), @roberthatch and AutoML Masters (@innixma et al.) were going to stand the test of time - since @alexrhyzkov & @roberthatch had generously shared their OOFs, I decided that would be my starting point. I also felt that a public LB score close to 72000 (without any blind blending) would probably fare well. The most consequential thing I did in this phase was to throw the 24 OOFs from @alexrhyzkov into AutoGluon, and confirm that they scored 72046/72057/72063 using GPU-P100/CPU/GPU-T4, respectively.

<b> Phase 2: Gradually intensive participation </b>
Around the 20th, I finally started putting OOFs together. I started by combining OOFs from @alexrhyzkov, @roberthatch & the eventual winner @martinapreusse - these 30 OOFs along with Ridge regression gave me 71958 on the public LB, and it scored 63001 on the private LB, which would have placed 5th. With my first sub-72000 score, I felt like things were on a firm footing, and from there, I went on adding a few models at a time. Given how little time I had, I only got 41 OOFs with 3 days to go - adding a model a day or so. Towards the end, I started adding new OOFs with a vengeance, playing around with whether to include the original dataset or not, changing hyperparameters, using AG with CPU/P100/T4, etc. Only on the second last day did I realize that the KaggleX dataset (generated using a GAN on the same original dataset, and hence likely quite similar to the competition dataset) was in fact accessible, and I quickly reran several notebooks with the KaggleX included. In retrospect, this helped with the public LB and not the private LB. Every now and then, I'd blend with the top scoring public notebook, and this usually improved the public LB, right from a 99:1 ratio (mine:public) down to 70:30 (but seldom beyond that).

In the end, I was up to 77 OOFs, and could have added more, but I could see that the CV score was barely improving, or getting worse occasionally. In the end, I chose my best scoring submission achieved without blending with another notebook's submission (71808, private 62958) and the submission with my best public LB score, achieved by blending the former with the highest scoring public notebook (71770, private 62977). All month long, we'd been debating whether there'd be a big shakeup, and there was indeed a massive shakeup, with everyone in the Top 11 besides me moving up 300 ranks or more! I was of course very happy to survive the shakeup, and get my 6th Top 10 finish in the last 7 playground competitions. A big congratulations to @martinapreusse, @gerlandore, @tilii7 and others in the Top 10 (and Top 1% etc.) & beyond who stuck to their guns, and reaped the rewards!

So, all's well that ends well. But there was one little revelation still left ......

<b> Epilogue: A potential no. 1 solution and a no. 2 solution that I overlooked - was Hill Climbing giving me a hint? </b> 

Every once in a while, I toss in an OOF prediction produced by an ensemble into my collection - I'd discovered this idea in a writeup by a former winner of a playground series (I don't remember which one right now), and it tends to work (but can also lead to overfitting - so use your own judgement). This time, I used such OOFs from AG based on 30 OOFs, and Hill Climbing (HC) after 33 OOFs - they both seemed to help, albeit only a little. It's Kaggle, I'll take it.

But when I threw in the OOF prediction produced by AG based on 41 OOFs (OOF score: 72185, public LB: 71855), a curious thing happened - HC with both positive and negative weights allowed worked as usual, but HC with only positive weights allowed didn't add even a single other OOF to this one. I'd never seen this happen before, and wondered whether this was some sort of sign that this was as good as it gets. However, Ridge did produce a better CV and LB score, so on I continued.

After the competition ended, I took another look at my scores - by searching for '6295', I found that I had 5 scores better than my final score (62958), all between 62950 and 62955. Then I searched for '6296', and found that I another five scores between 62959 and 62966. '6294' didn't turn up anything, so I figured I had 11 scores between the scores that place 2 and 4, and had dinner. Afterwards, I realized there might be lower scores, and sure enough there was a 62933, a potential no. 2 score! This one was produced by Ridge using 48 OOFs, including the 41 OOF based AG-OOF. This spurred me to look further, and sure enough, there was a 62892 - and guess what, it was from the same run as the AG run with 41 OOFs mentioned above, except that that one was with a GPU (T4), while this was with the CPU(s) (OOF score: 72092, public LB: 71922). So I missed out on a potential no. 1 score. Can't feel bad though, because I wasn't likely to select that, even if I'd stuck with only my own ensembles.

<b>PPS : Ridge produced the best ensemble scores</b>
For a while, I used various ensemblers - Ridge regression, AG, HC with and without negative weights (didn't use GBDTs this time).  As an example, these were the results using 30 OOFs with various ensembling approaches:

| Ensembler |CV score  | Public LB| Private LB|
| --- | --- |
| Ridge Regression| 72018 | 71958| 63001|
| Autogluon, GPU T4 | 72311 | 71993| 63029 |
| Autogluon, GPU P100 | 72317 | 72000| 63028 |
| Autogluon, CPUs | 72334 | 72011| 63029 |
| Hill Climbing| 72297 | 71973 | 63007 |
| Hill Climbing, positive weights only| 72321 | 71984 | 62991 |


This is fairly representative (except that varying the CPU/GPU choice didn't produce AG results in any fixed order). Ridge almost always produced the best CV and LB scores, and though we saw that it occasionally didn't produce the best private LB score, it's blazing quick, and you can't go far wrong with it. Which brings us back to

<b> Gather, Ridge, Repeat. </b>
