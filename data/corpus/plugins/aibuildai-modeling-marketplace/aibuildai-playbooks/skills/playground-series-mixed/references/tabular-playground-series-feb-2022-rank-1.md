# #1 Solution: Exploiting the Flawed Random Generation

Competition: tabular-playground-series-feb-2022
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-feb-2022/discussion/310359

### Observations

My solution is based on three observations about this TPS competition:

1. Train and test distributions differ, a fact that was discovered early in February during EDA. The difference of the distributions was intended by the paper authors who wanted to train their algorithms on ten bacteria and then test on ten slightly different (mutated) bacteria. The difference led many competition participants to "postprocess" the predictions of their classifiers.

2. If the cross-validation score is computed correctly (i.e. without duplicated data in the validation set), ExtraTreesClassifier reaches a public lb score which is substantially higher than the cv score. This observation (public lb score > cv score) suggests that there exists a leak between training and testing data. The main challenge is finding this leak and exploiting it.

3. A third of the training data and a quarter of the test data are duplicates. @siukeitin documented that these duplicates are the result of a strange random generation process (see the [source code](https://www.kaggle.com/siukeitin/tps022022-what-is-causing-these-duplicates-csi)). The observation that several hundred training rows are duplicated in the test data, however, got less attention. 

It turns out that the duplication between train and test can be explained through the random generation process. Beyond the duplicates there are many more almost-duplicates, and the ExtraTreesClassifier reaches its high public lb score exactly because of these almost-duplicates.


### The flaw in the random generation process

The following four lines of code demonstrate the flawed random generation. Let us generate two arrays of 30 digits in the range from 0 through 9. The `choice()` function requires an array of ten probabilities as input. We call `choice()` twice with different probability arrays, but with the same seed:

```
prob_train = [0.10, 0.10, 0.10, 0.10, 0.1, 0.10, 0.08, 0.12, 0.10, 0.10]
prob_test  = [0.12, 0.09, 0.09, 0.15, 0.1, 0.05, 0.10, 0.10, 0.09, 0.11]

print('Train:', np.random.RandomState(seed=231).choice(10, size=30,
      p=prob_train))
print('Test: ', np.random.RandomState(seed=231).choice(10, size=30,
      p=prob_test))

Train: [7 5 4 3 8 9 0 9 0 1 0 4 7 4 0 0 4 2 8 1 7 2 9 8 6 7 0 7 9 4]
Test:  [7 5 3 3 8 9 0 9 0 1 0 4 7 4 0 0 3 2 8 1 7 2 9 8 6 7 0 7 9 3]
```

You can see that the two random arrays have almost the same content except for three positions where the training array has a 4 and the test array has a 3. Why is this the case? The answer can be found in the [source code of the np.random.RandomState.choice function](https://github.com/numpy/numpy/blob/4a8007d5d916126965e811cd1b41ff4de44663b3/numpy/random/mtrand.pyx#L954-L957). To generate 30 random digits, the function first generates 30 random floats between 0 and 1 and then maps them to p.cumsum(). If we apply a small change to p, we'll only get a small change in the output:

```
cdf = p.cumsum()
uniform_samples = self.random_sample(shape)
idx = cdf.searchsorted(uniform_samples, side='right')
```

That's it. Train and test data are not independent. It happens that ExtraTreesClassifier profits from this leak, but a purpose-built classification pipeline performs even better. The challenge lies in finding a data transformation (and a suitable metric) such that data points generated with similar p and the same seed become close neighbors.


### The solution

My solution to the competition consists of a data transformation which makes the paired train and test points visible and a RadiusNeighborsClassifier which matches corresponding rows of train and test. For the low-resolution samples (those with 100 decamers), the transformation takes the 286 decamer counts of every row and converts them back into the list of 100 decamers which was the output of `np.random.RandomState.choice()`. The diagram below shows the transformed data with many close pairs of a training point (blue) and a test point (yellow). Details of the implementation can be found in [this notebook](https://www.kaggle.com/ambrosm/tpsfeb22-exploiting-the-flawed-random-generation).

[transformed]

I am quite certain that the biologists who gave the idea for the competition didn't want to produce a dependence between train and test. The result of their [paper](https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full) might look different if the data had been sampled correctly.

### It has happened before

In Section 3.1 of Volume 2 of his [*Art of Computer Programming*](https://seriouscomputerist.atariverse.com/media/pdf/book/Art%20of%20Computer%20Programming%20-%20Volume%202%20(Seminumerical%20Algorithms).pdf), [Donald Knuth](https://en.wikipedia.org/wiki/Donald_Knuth) describes how he attempted to create a fantastically good random generator using an algorithm called "Algorithm K". When this algorithm was first put onto a computer, it almost immediately converged to a fixed point. Knuth concludes with the moral that "*random numbers should not be generated with a method chosen at random*. Some theory should be used."

I have nothing to add to Knuth's conclusion.
