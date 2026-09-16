# 2nd place solution

Competition: abstraction-and-reasoning-challenge
Rank: #2
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154391

First of all, congratulations to @icecuber. Scoring 0.79 on this competition going solo is really impressive. Also, thanks a lot to @fchollet for hosting this competition and making us face this very interesting challenge.

Here I want to briefly discuss the alrogithm that @rguigocoro and I have developed. It scores 0.87 or 0.88 on the private test set I think. The final score of 0.81 on the LB came as a result of merging teams with @yujiariyasu, who had done a great job in solving many tasks already, and we luckily had almost no overlap. His approach is explained [here](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/138548?fbclid=IwAR0rVWjYYPzF3hDFT109Jsp-ezHsqttJzsVBgnVyG4JrFRUH7tBPIlia4y0).


For every task, our algorithm follows these steps:
1. Generate an object of the class `Task`. At this step, all the preprocessing is done: checking for matrix shapes, common colors, symmetries... There are 6 core classes that are used to store information about the preprocessing: The classes `Task`, `Sample`, `Matrix`, `Shape`, `Grid` and `Frontier`.
2. Based on the information retrieved from the preprocessing, the task might be transformed into something that can be processed more easily. It might make sense to modify some colors, rotate some of the matrices, crop the background or ignore a grid, for example.
3. Once the transformation is done, we generate 3 dummy candidates and enter the loop to get the 3 best candidates. In every iteration of the loop, for each of the current best 3 candidates, we try to execute all sorts of operations that make sense to try, given the properties stored in the object of the class `Task`. Many different sorts of functions are tried, such as "moveShapes", "replicateShapes", "extendColor" or "pixelwiseXorInGridSubmatrices", just to mention a few examples. The complete list of functions can be found on the GitHub repository linked below. If any function generates a candidate with a better score than any of the current best 3 candidates, we include this new candidate into our list of best 3 candidates, and remove the worst one. The score is simply computed by executing the functions that led to generate that candidate in the training samples, and checking how many pixels are incorrect. Thus, a score of 0 is the best possible one.
4. Revert the operations executed in step to 2 from the 3 candidates, to obtain the final candidates.
5. If it makes sense, we generate another task imposing that in every sample there can be either only one shape or only one non-common color. Obviously, we then have more training and test samples than in the original task. Then, we also compute the three best candidates for this version, and compare them with the previously generated ones to select the final best candidates.

It was quite hard work to set up this whole pipeline, and the first 1.5 months or so, being unable to go under 0.99, were a bit frustrating. But in the end this has proved to work reasonably well.
We try lots of different functions in our algorithm, and we don't really know which ones solve tasks from the private test set, as many of our LB improvements came after general preprocessing steps that are applicable to many different kinds of tasks.

All the code, including the version that scores 0.813 on the leaderboard, is on [this](https://github.com/alejandrodemiquel/ARC_Kaggle) GitHub repository.
