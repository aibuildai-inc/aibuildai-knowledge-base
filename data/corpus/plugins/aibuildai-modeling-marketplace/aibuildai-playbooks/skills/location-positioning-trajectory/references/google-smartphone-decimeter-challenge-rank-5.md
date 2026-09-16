# 5th place solution (the GPS data part)

Competition: google-smartphone-decimeter-challenge
Rank: #5
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/261746

Thanks to the hosts for this competition!

I knew nothing really about GPS prior to this competition, so it was a lot of fun (though also often frustrating!) learning all about it.



Our team had three main parts to the solution:

1. GPS baseline and relative position
2. Post-processing
3. ensembling / averaging

My primary focus was to improve #1 - the baseline lat/long points and a delta lat/long from one point to the next using the raw GPS information.

First, I'd like to say that there were a lot of pitfalls and dangers in the data! It was rather difficult (since I'm not a GPS expert at all) to really understand the data, and even after I understood it, the data was often a bit messy (with missing or skipped values), and it was tricky to really get everything working.

Also, can I just say: converting between UTC and GPS time is no fun! 😂 Add in the rotation of the earth between the signal transmit and receive time, converting between ECEF, ENU, lat/long, and that all meant a lot of debugging sessions trying to figure out exactly where I went wrong... 


Here are several things I did that worked, and some that didn't.


# Filter out obviously bad pseudo ranges

I used the same filtering criteria as the host's baseline guidelines: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/238583

That seemed to work fairly well to remove most pseudo ranges that were obviously bad



# Improve Satellite x,y,z positions

The _derived files contained satellite x,y,z positions, but the contest also included satellite correction information from SwiftNav.  I was able to apply the SwiftNav corrections to more accurate satellite x,y,z positions, but that ultimately didn't have much effect on the GPS baseline (because the given positions were already pretty close)


# Using carrier phase / AccumulatedDeltaRangeMeters

To smooth the very jumpy pseudo ranges, I used AccumulatedDeltaRangeMeters to calculate a delta pseudo range for every satellite for every point. Then, I could calculate a smoothed, or average pseudo range for each point and use that.

This was fairly effective, since the AccumulatedDeltaRangeMeters is much more accurate than the raw pseudo range measurement


# Using doppler / PseudorangeRateMetersPerSecond

Whenever the carrier phase cycled (which was often - especially in SJC), I used PseudorangeRateMetersPerSecond to approximate the AccumulatedDeltaRangeMeters between points. This worked well for a few seconds at a time, but more than 10 or so seconds became a problem, since PseudorangeRateMetersPerSecond error rates were higher than AccumulatedDeltaRangeMeters.

Also using doppler, I was able to calculate a relative, or delta lat/long between each point by taking the difference between the measured pseudo ranges and the expected difference given by either the doppler or carrier phase.

That delta lat/long measurement was later used in post processing.


# Adjusting the satellite uncertainty

I noticed that the GPS and GAL satellites were more accurate for most phones, especially when surrounded by tall buildings (I'm not 100% sure of the reason behind that).

To account for that I tried a variety of things, from excluding other satellites entirely, to just multiplying the uncertainty by a factor of 2, 4, or 10. That seemed to work well for this dataset, but may not be valid for other geographies.

Once we had some good post processed data points, I also increased the uncertainty for pseudo ranges that were far outside of that range. 


# Reducing multipath

I also assumed that pseudo ranges that were too "long" (compared to the current best known location) had a higher chance of being affected by multipath (because they could have reflected off of buildings or cars). (Not being a GPS expert, I actually don't know if this is correct at all :) but it was a guess).

To try to account for that, I experimented with removing the top X% of pseudo ranges (when sorted by expected residual error), or increasing their uncertainty. Removing them actually improved downtown areas quite a bit.


# Other ways to reduce the impact of bad pseudo ranges

I tried other off the wall ideas to reduce the impact of _really_ bad pseudo ranges, some of which had a small effect, but none of which were very noticeable.  

For example: I tried taking both the square root and the log of the (measured - pseudo range) value in an attempt to reduce the large errors (before putting it into least squares). That actually DID work, but the result was a bit inconsistent between collections, and wasn't a very large effect on most. 

I do think there is more work that could be done there though.



# Differential or relative positioning

If there was one thing I would have liked to have spent more time on, it would be differential or relative positioning based on known GPS reference points.  I tried this several times - both with the given SwiftNav reference point and others, but I just couldn't get it to work.

Part of the problem is the gps receivers in the phones were just very noisy - it seemed that a lot (if not MOST) of the error came from receiver noise, which means both differential and relative positioning (including double difference and triple difference) would actually INCREASE that error (if I understand correctly).

Another rather annoying thing was that the phone receivers were recording on a fraction of a second (e.g., 0.44) rather than right on the GPS second (like the reference receivers). That made it much more difficult to even attempt differential or relative positioning.

I was never able to get this working completely, and I think it's probably the biggest thing I could have done differently.



# "self" relative positioning

One thing that I did get working but that didn't actually improve the baseline like I expected was doing relative GPS positioning with each phone to itself - that is, to other positions in that phone's path.

I was assuming (hoping) that the receiver error was approximately normally distributed around the correct answer, and thought that doing relative positioning with (for example), 10 seconds of other points around each point would even out that error.

I _kind of_ got this working, but I could never get it to produce the results I thought it should... maybe if there's a new GPS contest I can figure out why that is :)


# Incorporating IMU data

I really wanted to incorporate the IMU data into the pseudo range averaging (with a Kalman filter, or even just a regular weighted average) - but I just ran out of time.

I think this would have greatly reduced the receiver noise that I saw, which should mean a much better baseline solution.


# Overall

I didn't find any magic bullets in the GPS baseline calculations

It took a LONG time to really figure out all the data, and what I was doing

The data was pretty noisy, and a bit difficult to work with - but I think a lot of that was because I've never worked with GPS data before.

I'd love to give it another try (now that I actually know what I'm doing) if there's a new competition sometime!
