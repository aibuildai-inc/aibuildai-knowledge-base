# How I got my run time down from 10 hours to 10 minutes

Competition: helping-santas-helpers
Rank: #20
Source: https://www.kaggle.com/c/helping-santas-helpers/discussion/11452

<p>Earlier in this forum there was a post about run time to process the 10 million toys. &nbsp; When I read it I was shocked to see that there were people in there talking about run times measured in minutes, because at the time my python solution implementation was taking 10-12 hours to run. &nbsp; However since I stumbled on how to reduce the run time significantly, I wanted to share for anyone else interested</p>

<p>At the time I was exploring 2 different solutions, one was algorithmic, and one was an evolved solution, and I was trying to get the evolved solution to run a lot faster. &nbsp; Both solutions involved reading in all 10 million toys, or as many as my computers RAM could hold without breaking. &nbsp;I played around with various code changes such as changing the get sanctioned, unsanctioned hours breakdown, which was counting the minutes 1 at a time, to an algorithm, and using faster ways of slicing my lists, but my runs were still hours.</p>

<p>The real thing that sped up my runs was changing my data structure. &nbsp; Instead of reading in all 10 million toys and processing them, I instead made a histogram of the toys. &nbsp; i.e. there were &nbsp;~ 100,000 &nbsp;one minute toys and ~90,000 2 minute toys etc. &nbsp; &nbsp;So I could store value 100,000 in list index [1] and 90,000 in list index [2] and process that. &nbsp; Suddenly instead of 10 million unique values, my computer was dealing with around 25 thousand distinct values which only took a couple of minutes, &nbsp; and then several more minutes to go in an back substitute actual toys for their durations</p>

<p>This didn't take into account relative start times, so I just started them all at year 2015, giving up a couple of months in exchange for easier coding</p>

<p>I should caveat to say that I did eventually abandon this solution in favor of my teammate's implementation which took longer to run, but for other reasons gave a better solution and was better structured to roll in improvements.</p>

<p>For others who had shorter runtimes, did you implement a similar solution? &nbsp; Or was there another way to speed up the code ?</p>
