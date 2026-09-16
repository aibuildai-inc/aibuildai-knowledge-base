# 30th place solution. Affine transformation and voting deepsort cluster

Competition: nfl-health-and-safety-helmet-assignment
Rank: #30
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285127

Congratulations to everyone for your great results, specially the winners. It has been a very consuming time and funny competition!

The solutions are really great and impressive. I have little to add with value, but I wanted to share this solution for, I think, its simplicity. Just using boxes offered and strandard DeepSORT.

It uses DeepSORT clusters and vote assignments by using affine transformation matrix moving coordinates from tracking data to video image coordinates.

### The pipeline
- Apply DeepSORT to offered boxes.
- Then on every frame:
 - Obtain the affine transformation matrix (least-squares minimization using 4 ground truth points). 
 - Linear sum assignment
 - Vote for cluster
 - As cluster receives enough votes, assign label.
 - Split clusters that after some time receives enough votes for another label 

### Obtaining affine transformation matrix
To obtain the affine transform matrix in the first frame it used 4 players-boxes combinations as ground-truth. Used the most outer boxes of the images (left, top, bottom and right) and tried using the 6 outer players on each side and assign a box as ground truth and compare the assigning costs on each combination.
- Get outer boxes (left, top, bottom and right)
- Iterate over:
 - Try a combinations of outer players (tracking data)
 - Compute transformation on all players
 - Linear sum assignment and get the cost of assignments
- Get the transfomation matrix with minimum assignment cost.

The score on the 20 first frames where in most cases 100%. The following frames the transformation matrix was calculated using the most isolated boxes/players. 
(Image: The little squares next to the boxes are the transformation result of the tracking data)

[[Initial-Assigment2.png]](https://postimg.cc/QHQKWYjg)

When advancing in the play the solution had problems mainly with 'z', when players where in the ground and assigning cluster with mixed players. But it had the ability to recover and get good scores again. (Like the image below recovering 100% of accuracy on frame 302)

[[midplay.png]](https://postimg.cc/v1B4kQfX)

SCORE
Position: 30th
Public LB: 0.814
Private LB: 0.732
