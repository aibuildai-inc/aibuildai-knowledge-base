# Hints (publicly sharing)

Competition: draper-satellite-image-chronology
Rank: #4
Source: https://www.kaggle.com/c/draper-satellite-image-chronology/discussion/21822#125285

Here's a quick outline of our approach which almost got us a perfect score. It sounds similar to that of others that accepted an approach that included handlabelling.

We split the task into four parts

 1. Geocode images
 2. Identify common days
 3. Order days
 4. Manually peer review and correct submission

Handlabelling featured in all parts of the above but we found python and R scripts helped greatly with the image registration and identifying cases where humans had made mistakes in steps 2 and 3.

Particular learnings included

 - we used shadow angles and intensities for our first pass of allocating images to common days
 - this worked particularly well for clusters where flights were at distinct and well separated times of day
 - where images in a set shared shadow angle and intensity we sometimes could differentiate and allocate to common days by comparing shadow lengths
 - for the really difficult images (ie 2 images from each set showing scrubland north of the landfill) where shadow angles and lengths could not be determined we found that images could be allocated to common days according to the rotation of the image itself relative to true north.

When it came to ordering days, we could usually find a number of image sets in each cluster that involved some sort of construction which physically progressed in each day, hence giving away the image order. We found that cars in carparks,  puddles, containers and rubbish in sorting centres etc were generally less reliable.

What made this competition "easier" was the fact that relative few distinct flights were made and so images overlapped, spatially and temporally. This meant it was possible to register images across sets to to the same days. Once you had determined the order of one set you then knew the order of all images in associated sets.
