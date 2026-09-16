# Hints (publicly sharing)

Competition: draper-satellite-image-chronology
Rank: #7
Source: https://www.kaggle.com/c/draper-satellite-image-chronology/discussion/21822#126701

I found my mistake.  I had a typo in set 225.  The effect of said typo was a -1.6 difference in spearman r correlation for what I actually entered for set 225 which when divided by 139 results in a private leaderboard delta of 0.01151.  That difference meant that I went from 6th on the public leaderboard to 7th on the private; 0.98849 on the private leaderboard instead of 1.0000.  So here's my first hint.  Beware of typos in a labeling competition...  I suppose this is where I should be grateful to all of the people who were faster than me; even without the typo I wouldn't have been in the money.  (I imagine 4th to be the most uncomfortable position in many Kaggle competitions.)

Here's my next hint.  There were 28 sets in the public leaderboard and 139 in the private.  I never did managed to order a set manually.  I did all of my ordering via the public leaderboard.  The sets in the public leaderboard are 

 - 330, 328, 215, 119, 275, 72 from La Jolla
 - 13, 259, 179, 38 from Logan Heights
 - 83, 145, 39, 234, 195, 7, 266, 240 from the Point Loma/San Diego Bay/Airport/Old Town area
 - 1, 111, 208, 254, 187 from Scripps Ranch 
 - 253, 152, 200, 67, 54 from Sycamore Landfill and scrub east of landfill

The possible values for Spearman's correlation coefficient between two orderings of 1,2,3,4,5 are 

    [1 - .1*x for x in range(21)] = [1.0, 0.9, 0.8, ..., -.8, -.9, -1.0]

If you know that a set is in the public leaderboard, you can find it's ordering with two reasonably chosen entries.  While I did have the dubious distinction of the most entries of any team in the competition, that is still not enough entries to isolate the leaderboard individually.  However, as mentioned in many places [\[see the winner's post\]][1], many of the pictures overlap.  So by finding groupings of pictures that were taken on the same pass of the same flight, you can triangulate the group if there is at least one member of the group in the public leaderboard using a combination of linear programming and logic.  For example

    Flight: 53, Tree like branching rds south of Spring Canyon rd through dome bldg to gravel pit  setIds: [284, 231, 88, 208, 333, 48, 233, 138, 172, 23]
      Day  set284  set231  set88  set208  set333  set48  set233  set138  set172  set23
    2   c       3       2      1       2       3      1       3       1       3      4
    4   e       5       4      2       4       4      4       5       3       5      3
    0   a       1       5      3       5       5      3       4       5       4      1
    3   d       4       1      5       3       1      5       2       4       1      2
    1   b       2       3      4       1       2      2       1       2       2      5

is one of the groupings through Scripps Ranch with pictures that are adjacent and overlapping and it happens to contain set 208 from the public leaderboard.  By entering groups of approximately 10 permutation matched sets at at time, I was able to use the public leaderboard to find the correct ordering for most groupings.

I did not match all of these by hand.  This leads to my next hint:  [template matching][2].  Take two sets that overlap and for each image in the first crop out a reasonably sized part of the overlap and use that as a template to match with each of the 5 images in the second set.  For each pairing let the score be the maximum correlation coefficient. For example looking at the scores from matching set333 to set48 using a crop window of `(1000,0,2000,500)`

    set333_1, set48_1   0.38268 
    set333_1, set48_2   0.40771 
    set333_1, set48_3   0.32304 
    set333_1, set48_4   0.42695 
    set333_1, set48_5   0.96155 
     
    set333_2, set48_1   0.28035 
    set333_2, set48_2   0.70019 
    set333_2, set48_3   0.30732 
    set333_2, set48_4   0.32623 
    set333_2, set48_5   0.31640 
     
    set333_3, set48_1   0.73081 
    set333_3, set48_2   0.51668 
    set333_3, set48_3   0.43541 
    set333_3, set48_4   0.45769 
    set333_3, set48_5   0.50292 
     
    set333_4, set48_1   0.19761 
    set333_4, set48_2   0.19939 
    set333_4, set48_3   0.17685 
    set333_4, set48_4   0.60496 
    set333_4, set48_5   0.21628 
     
    set333_5, set48_1   0.38741 
    set333_5, set48_2   0.49584 
    set333_5, set48_3   0.96844 
    set333_5, set48_4   0.46904 
    set333_5, set48_5   0.47248 

we see that set333_1 and set48_5 were probably taken on the same pass, etc.  Template matching does lead to weird discoveries like the fact that even though sets 30 and 145 look like they overlap, they are not taken in the same pass of the airplane.  

Template matching breaks down because not all of the pictures taken in the same pass overlap another picture in that pass.  Also water exposes a bug/feature in scikit image's match_template code. So as a second method and my next hint is shadows.  Shadows were very useful for connecting both sets that didn't overlap other sets and flight groups that didn't have any public leaderboard presence to flight groups that did.  

As a final hint, beware of rotation matching.  It works much better on the training set than the test set.

By treating this whole competition as a crazy combination of Where's Waldo and a jig saw puzzle, I didn't actually have to resort to external data although I must admit to using google maps to help confirm which pictures belonged in which area.  (It was the picture on the box of the jigsaw puzzle...)  In post competition checks I've confirmed that none of the LA pictures were in the private leaderboard.  I started this as an exercise in trying to increase the amount of training data to something that would let me build a model...  and never did get to building a model.



  [1]: https://www.kaggle.com/c/draper-satellite-image-chronology/forums/t/21936/1st-place-how-to-win-the-competition-if-you-know-nothing-about-image-processing/125346#post125346
  [2]: http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=match_template#skimage.feature.match_template
