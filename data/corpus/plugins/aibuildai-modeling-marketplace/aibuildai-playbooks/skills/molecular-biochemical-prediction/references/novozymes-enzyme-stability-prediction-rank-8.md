# 8-th place solution - AVGP3 + Rosetta

Competition: novozymes-enzyme-stability-prediction
Rank: #8
Source: https://www.kaggle.com/c/novozymes-enzyme-stability-prediction/discussion/376070

First of all, we are very happy to be a part of such unordinary and interesting competition, many thanks to the organizers and competitors who selflessly shared their knowledge during this competition, @vslaykovsky, @oxzplvifi, @dschettler8845, @kvigly55, @cdeotte, @shlomoron …, If I've missed someone, please let me know.

Many thanks to my great team QuData @synset, @alexz0, @kerrit and @semenb, you are awesome! Despite the constant blackouts in Ukraine and the explosion of rockets over our heads, we enjoyed diving into the world of biochemistry.

## AVGP3

[Thermonet features]

At the beginning of the competition, we noticed that the https://www.kaggle.com/code/vslaykovsky/nesp-thermonet approach gives a high contribution to the prediction of thermal stability.

We studied the material from this article https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008291 and decided to experiment with different neural network architectures, and arrived at the Resnet-3D with a public score 0.488.

Then we wondered which parts of the network really affect the result. By iteratively pruning the network we came up with a fairly simple architecture:

```py
def forward(self, x): 
    x = x.mean(axis=(2,3,4))
    return -x[:,3]<br>
```
The same shape as for the Thermonet work was used as the input, (14,16,16,16).

This model gave a metric of 0.477 on public lb. Thus, averaging a single parameter **hbond_donor** (hydrogen bond donor) yielded a fairly high result.

To improve the metrics, we built different variations of feature descriptors using the **molekulekit** utility. The optimal ones were: boxsize:16, voxelsize: 0.5, which improved public lb to 0.482. This model, which we called AVGP3, became a part of our final ensemble solution.

## Rosetta

Thanks to the notebook https://www.kaggle.com/code/oxzplvifi/deletion-specific-ensemble we noticed that thermal stability is also well predicted by Rosetta scores.

Combining two methods in the ensemble (0.5 rosetta + 0.5 AVGP3) we got **0.590** in public lb and, as it eventually turned out, **0.540** in private lb (**4th** place). But unfortunately we didn't choose this result in the final submission.

Next, the AVGP3 result was combined with https://www.kaggle.com/code/oxzplvifi/deletion-specific-ensemble with weights of 0.2 and 0.8 and improved our public lb score to 0.613

We then proceeded to examine what Rosetta Scores success consists of. After the relaxation procedure, the pdb files contain POSE_ENERGIES_TABLE at the bottom

[energy table]

which consists of 20 parameters and weights to them, and we used them in our approach.

## Ranked rosetta

The idea emerged to use the weights for the ensemble. Independent predictions of thermostability were created for each parameter separately for all mutations, and then combined into an ensemble using the weights from POSE_ENERGIES_TABLE 

In fact, we have replaced:
$$ tm = {rank(\sum_{i=1}^n w_i * prop_i)} $$
on:
$$ tm = {\sum_{i=1}^n w_i * rank(prop_i)} $$

This ensemble gave a score of 0.338 (we call it RankedRosetta)

Combining this result with the best ensemble (0.9 * 0.613 + 0.1 * 0.338) we got 0.615 on the public lb.

## Final ensemble

On the final day, after analyzing the forums, another result (https://www.kaggle.com/code/sgreiner/novo-esp-rasp-baseline) was added to the ensemble, which added another point to our public score
(0.05 * Rasp + 0.95 * 0.615) = 0.616 and gave us the 0.532 private score.

So, to sum it all up,

the best submitted result:

0.95(0.9(0.8 (DelSpec:0.603) + 0.2 (AVGP3)) + 0.1(RankedRosetta)) + 0.05*(RASP) = 0.612 pub, 0.532 private (**8th** place)

the best result overall (non-submitted):
0.5(Rosetta)+0.5(AVGP3) = 0.590 pub, 0.540 private (would be **4th** place)

As we can see, the thermostability in both cases was significantly affected by Rosetta Scores throughout the molecule and the average of the hydrogen bond donor parameter in the vicinity of the mutation.

Thanks to all!
