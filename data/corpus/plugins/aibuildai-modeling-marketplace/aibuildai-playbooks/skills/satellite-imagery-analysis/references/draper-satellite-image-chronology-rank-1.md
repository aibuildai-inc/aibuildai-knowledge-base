# 1st place - How to win the competition if you know nothing about image processing

Competition: draper-satellite-image-chronology
Rank: #1
Source: https://www.kaggle.com/c/draper-satellite-image-chronology/discussion/21936

For this competition I trained the neural network (the one inside my skull). 

It took about 2 weeks of staring at the images until the magic happened.

Here are the main ideas.


High level algorithm:
---------------------


 1. Split the data by location

 2. For each location match together the pictures that were taken in the same day

 3. Solve each location using any clues from any picture set within that location


Ad 1:
-----

We know that the pictures were taken in the southern California and on the pictures we see ports, airports, landfill, golf field, Point Loma Stadium ...

So using Google Maps it is relatively easy to find all the locations:

Train set: Carson (LA County), Port of Long Beach (LA County)

Test set: San Diego (near downtown), Point Loma (W peninsula of SD), La Jolla (N of SD), Sycamore Landfill (E of San Diego/Santee), Sycamore Estates (N of landfill)

It is not necessary to point the exact locations of the set, discovering the locations that were taken within one flight is sufficient. (I think that Sycamore Landfill + Sycamore Estates were taken within one flight, similarly San Diego + Point Loma and Carson + Long Beach, so there might be only 4 locations if we merge these together).


Ad 2:
-----

The idea is that the plane flew into the area, took all the pictures within a couple of minutes and then flew away.
Thus if the pictures were taken in the morning, all the shadows should point to the left, if it was evening then they should point to the right, and during the noon to the top.
If it was cloudy during the flight over that location we should see only weak shadows (there was one such a day over the whole San Diego area, that was very easy to spot).
If we look at the map with marked locations, we see that the plane followed the straight lines when it took the pictures, so the neighboring images within that line should have very very similar orientation and zoom and in many cases also some overlap.

Overlaps and shadows were most important features in this step (shadows were used mainly in cities because they very easier to evaluate by hand + they could bind the areas that didn't overlap; overlaps were the most reliable, but it took a little bit longer to match the pictures by hand using overlap than using shadows, they were mainly used in forests when the shadows very hardly visible, and when the two shadows seemed very similar).

In some cases we could also use secondary features like presence/absense of cars at some location (indicator of the weekend), presense of clouds, presense of puddles after a rain. In the train set the zoom was an important feature.


Ad 3:
-----

In this step we could use any clue from any set from the location to solve the whole location.

Building sites, tracks in the mud, piles of sand were the most important and reliable features.

Presence/absence of the cars in from of hypermarkets (weekend), presence of the cars in front of church (Sunday), presence of puddles after a rain (Point Loma) were also very important.

Parking patterns of personal cars were not very reliable, because people use them every day and they may have a reserved or preffered parking place and the presence of the car at the very same place might not be a strong indicator of the closeness of that days. However the parking patterns of the bigger trucks and boats were pretty reliable.
