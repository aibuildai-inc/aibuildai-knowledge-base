# Santa25 8th Place

Competition: santa-2025
Rank: #8
Source: https://www.kaggle.com/c/santa-2025/writeups/santa25-8th-place

# Santa25 8th place


## Introduction

First of all, I would like to thank the organizers for making this competition possible. I would also like to thank the rest of the participants for their ideas, and especially the creators of spyrrow/sparrow. You can find my submission.csv at the end, on "Project Files"

## Outline of my solution:

• A web interface to visualize, select, and manipulate the trees

• Design of different tile patterns to cover most of the solution space for large N.

• Use of spyrrow (unmodified) for small N and to fill in the remaining cases for large N.

• Development of a tool to compute the exact outline/profile of arbitrarily shaped regions.

• Use of simulated annealing (SA) to refine the solutions.

## Hardware and pipeline:

I used my laptop to manage the graphical interface, connect to the other machines, and manage the processes. On two machines with 128 threads, spyrrow was running continuously over two separate queues of cases to simulate. The results of these simulations were sent to another machine with 24 threads, which was responsible for managing the queues to run the simulated annealing (SA).

The workflow involved progressively debugging the different configurations, selecting subsets of them, computing their exact profile, and instructing spyrrow to place the remaining trees—sometimes adding or removing trees in order to export configurations of N trees to N ± n (this often made it possible to change the initial tiling).

## Web interface

The web interface allowed me to view the configurations, compare one against another using the two windows, select trees, rotate them, delete them, etc. Above all, it helped me choose a candidate subset (due to having low local density) to remove and then refill using spyrrow.



## Design of different tile patterns 

I have invested quite a lot time designing tile patters with different criteria (density, aspect ratio, angles, base shape, ...) and used them as starting points of my solutions.


## Use of spyrrow (unmodified) for small N and to fill in the remaining cases for large N

In my experience, the solutions produced by spyrrow have a lot of variance, so it was useful to run many simulations because every now and then we get a nice outlier.

## Development of a tool to compute the exact outline/profile of arbitrarily shaped regions

I used the idea of selecting a subset of trees and treating it as a single shape for spyrrow, defined by its vertices. I initially used alphashape to define the outline/profile—by tuning the alpha parameter you can get better profiles—but I wasn’t fully satisfied with the results, 



so I developed my own profile evaluator for arbitrary shapes:



In this way, I obtained solutions like these that combine a tiling core with isolated trees around the periphery:



## Use of simulated annealing (SA) to refine the solutions

The final step in the pipeline was to use simulated annealing (SA) to refine the solutions: everything that came out of spyrrow was pushed into a queue and refined with SA (in fact, I had two queues with different SA parameter settings, and I moved solutions from one to the other based on certain criteria).
