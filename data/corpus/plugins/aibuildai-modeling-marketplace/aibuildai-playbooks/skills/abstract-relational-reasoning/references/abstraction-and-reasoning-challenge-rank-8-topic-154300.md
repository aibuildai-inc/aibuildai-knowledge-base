# 8th Place Solution (Part 1) - Object detection

Competition: abstraction-and-reasoning-challenge
Rank: #8
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154300

Thank you to Kaggle and Francois for hosting this awesome competition. 
Thank you to my teammates for doing an awesome job.
Thank you to @meaninglesslives for giving hope in this challenge by breaking the 1.00 submission and shareing it publicy.

Congratulations to every medalist in this competition. It got realy close in the 0.862 region and it was fun to see the top of the LB especially in the last hours in the race for price money.

Congratulations to @icecuber on winning this competition. 

As there are many different approaches needed in this competition our team decided to split our results in 3 parts.

Part 1: 
My Solution is about simple object detection which solves 6 tasks in public LB.
I put it together in a kernel (a bit messy - did not spend time yet on cleaning code, but added some visualisation which I hope help in understanding the code; feel free to ask any question):

 [https://www.kaggle.com/jpbremer/very-simple-dsl-solving-up-to-6-tasks](https://www.kaggle.com/jpbremer/very-simple-dsl-solving-up-to-6-tasks)

It solves 4 tasks in its current form. With some more properties/ luck it was able to solve 6 tasks in total on LB.

The pipeline is:

1. evaluate properties from the training outputs (e.g. how many colors in output, symetries, ...)
2. random crop input
3. check if crop has the same properties as the properties retrieved from training tasks
if yes: append
if no: repeat from step 2

I am looking forward to interesting approaches by different competitors and would like to see the test tasks. (especially the one nearly all public kernels solve). 

I am proud to become a Kaggle Master in this awesome community. 
Kudos to everyone involved in this competition!
