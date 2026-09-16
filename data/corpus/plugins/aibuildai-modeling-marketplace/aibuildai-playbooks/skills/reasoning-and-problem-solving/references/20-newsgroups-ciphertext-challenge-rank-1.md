# 1st place solution # c1ph3rh4ck team

Competition: 20-newsgroups-ciphertext-challenge
Rank: #1
Source: https://www.kaggle.com/c/20-newsgroups-ciphertext-challenge/discussion/77894

Hi, 

Thank you Phil for organizing this competition, it has been an addictive and intense experience! 

We learnt a lot in the process and wanted to share our experience in this post. This post does not cover all technical details and everything we tried, we mostly wanted to tell the story of this competition from our eyes.

And before we start, a massive thank you to our competitors, and special note for EtienneW, flal, kaggleuser58, RS Turley. You really pushed us to do better. THANK YOU.

## Episode 1: The c1ph3r h4ck1ng. From 0 to 0.9847.

I called my friend and I was like "hey there is this competition, are you up for it?". Minutes later, we jumped on a mission to break the ciphers one by one. We kept going until success. For a detailed technical write-up about the ciphers breaking, we would like to recommend the great kernels by our competitors listed above. It took us a few days, we used a mix of automation, stats and manual fixes to get it right and reach the top of the leaderboard. We also painfully reverse-engineered the pre-processing ("\n " vs "\n" pain anyone?).

However, we found some ambiguous rows. We submitted a random solution in the space of possible solutions. 0.9847. *Have we missed something?*

## Episode 2: The space fine-tuning. From 0.9847 to 0.9991.

We improved our solution:

1) We counted how many times each piece of text appears in each class.  For example instead of mapping the '--' decoded text to the potential classes [1, 5], we mapped it to {1: 1, 5: 4} (keys = classes, values = number of occurrences). Then instead of uniformly sampling on [1, 5], we gave **non-uniform probabilities** (or equivalently uniform ones on [1, 5, 5, 5, 5]). This technique gives a higher probability to the correct class, and tends to increase the score.

2) We grouped rows with identical decoded text. For example, if a decoded text is found twice in the test set, and that text maps to {1: 1, 5: 1} (classes 1 and 5 are both found once). Then we know the solution is

`row_1, 5`<br>
`row_2, 1`<br>
or <br>
`row_1, 1`<br>
`row_2, 5`<br>

In other words, using **sampling without replacement** for each group reduces the probability space.

3) We **removed the observations from the training space**. For example, if a string, '---' appears in classes 7 and 8, but is in class 7 in the training set, then we know class 8 is the one we want.

Jointly, this brought us up to 0.9991. Not bad. 

## Episode 3: The RNG gods incantations. From 0.9991 to 0.9992.

It was time to invoke the RNG gods (shoutout to Etienne for this expression ^^) with our tricks from Episode 2. Our incantations gave us solutions scoring in [0.9990, 0.9992]. We were not so happy to rely on luck at this point, but we were up +0.0001, so we'll take that.

## Episode 3: The RNG gods on steroids. From 0.9992 to 0.9994.

At this point, we had a bunch of random submissions, some scoring better than others. Despited limited precision, we decided to aggregate that information. 

1) We used a **simple counting technique**. For example, for a row with 2 possible classes {5, 6}, we gave points to each class. We decided to give points as: 0.9990 -&gt; -1, 0.9991 -&gt; 0, 0.9992 -&gt; +1. Iterating over our random submissions, we got, for example, 3 points for class 5, -1 point for class 6. Then we created a submission with all winning combinations.

2) We used **dimensionality reduction techniques** to somehow visualize the space we were sampling from. We sampled lots of random submissions, both for exploitation (submitting a solution close to our top scores in the 2D space) and exploration (submitting a solution which is far away from the ones already submitted -- hoping that it would provide useful information we would use at some point).

Both techniques were enough to stay at the top of the leaderboard for a while. However EtienneW was catching up, and he finally took the #1 spot. We kept using our techniques for one or two days, then decided it was time for a change.

## Episode 4: The mathematical formulation. From 0.9994 to 0.9994.

We were not excited by the idea of manually testing ambiguous rows one by one. Instead we preferred to look for an elegant way of finding the solution. So we started to wonder which information we had not exploited yet, and could be leveraged. At this point, we were a bit frustrated by the limited precision of the scores of our past submissions. We checked the HTML code and the Kaggle API, no luck, same limited precision. 

Looking for extra information, we found out that, on the Kaggle "My submissions" page, it is possible to sort past submissions according to their public scores. We hypothesized that, despite an apparent identical score, the submissions were sorted according to the full precision score. **This gives much more information than the scores only.** 

`some submission, 0.9994  # rank = 1` <br>
`another submission, 0.9994  # rank = 2`<br>
`...`<br>
`yet another submission, 0.9993  # rank = n`<br>
`...` 

The question was: how to leverage that information? Our first idea was to refine our point system from Episode 3. 

And then it hit us. **We realized it was possible to evaluate candidate solutions without submitting them.** Let me explain. If you have the correct solution, then you should be able to use it to score and rank your past submissions.  Let us say you have a candidate solution. Then if you compute the score of a past submission, assuming the candidate solution to be perfect, you should get the score visible on "My submissions", for any submission. In addition, given a candidate solution, if you compute all past submissions and sort them according to these scores, you should get the same ranking as the one visible on the "My submissions" page, when sorting by "public score". 

Of course, this is a **necessary but not sufficient condition**. Nevertheless, we could evaluate tons of solutions locally, and hopefully have an strong enough edge to win. Unfortunately, the space to bruteforce was just gigantic. 10^{39}. Even with a good implementation, we don't have that kind of computing power. So we decided to reduce the space size with manual submissions.

## Episode 5: The manual improvements. From 0.9994 to 0.9995.

As we could not solve the nice optimization problems yet, we had to reduce the number of possible combinations. With the "sort-by-public-score" technique, we could **evaluate the impact of a single change**. For example, we take one row where there are two possible solutions, say classes 7 and 8. We take our best submission, change the value on a single row, and submit this new solution. If this new submission gets a better rank, then the new value is the correct one, otherwise the old value is the correct one. In any case, 

 1. we get the information of the correct class for that row and,  
 2. we reduce the space size by a factor 2.

Not super exciting phase, we got +0.0001 and we were still #2. The space was still too large to fully bruteforce.

## Episode 6: The leaderboard shake-up. From 0.9995 to 0.9998.

At this moment, the leaderboard was taken by storm. Flal and kaggleuser58 scored 0.9997 (not sure who got there first). We were #4. Flal had only 4 submissions, so we knew he did something smart we missed. 

Again, the question was: which information could be exploited? We realized the `difficulty` field carried extra hidden information. **All chunks for the same original document were encoded with the same `difficulty`.** This reduces the number of ambiguous rows. For example, say some encrypted text is decoded as '--', which maps to classes 7 and 8. Now let's say we also know that 

 1. '--' was encrypted in difficulty 3, 
 2. one document with the chunk '--' was encrypted in difficulty 3 and is in class 7, 
 3. another document with the chunk '--' was encrypted in difficulty 2 and is in class 8, 

then, we know that the correct class is 7.

This is not enough to solve everything, but still it was a massive solution space reduction! We went from 128 ambiguous rows to 37 ambiguous rows. Unfortunately, the space was still too large (~ 10^{15}), but we were getting there. With our 5 daily submissions at hand, we solved 4 ambiguous rows manually (further reducing the space by 2^4). Finally we shot 6M random solutions in this space, and took the solution with the lowest scoring error, as explained in Episode 4. We were back to #1 with 0.9998.

## Episode 7: The final strike. From 0.9998 to 1.

From here on, we had a plan to finish this: **manual space reduction followed by bruteforcing** (our secret weapon). 

The space was still huge, so we started to look into code efficiency and speed-up tricks. Several orders of magnitude later, we were a few days away to fully bruteforce the space with our python implementation. Nevertheless, the competition was tough, so we re-implemented our strategy in C++. We gained further orders of magnitude. It took around 20-30 min to evaluate 5x10^{14} (50k billion) solutions on a 24-CPU VM.

We got 1280 solutions that were able to perfectly score all our past submissions. Not bad. Then we evaluated these guys in terms of ranking and found... no solution. We strongly believed in our approach, so with high probability there was a bug somewhere in our pipeline. The bug hunt was unsuccessful for a while, so in the meantime we submitted solutions with single row changes. This brought us to 0.9999.

After about two days of checking assumptions, reviewing code, checking everything, and lots of sweat, we finally fixed our pipeline, and found the perfect solution. Submitted that. Kaggle called Watson. Got that final +0.0001 ^^ 

**JUNG N PBZCRGVGVBA! GUNAX LBH NTNVA!** You know what we mean, right? ;)


Team c1ph3rh4ck.
