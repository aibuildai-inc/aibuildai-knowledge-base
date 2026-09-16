# 4th place solution

Competition: outbrain-click-prediction
Rank: #4
Source: https://www.kaggle.com/c/outbrain-click-prediction/discussion/27926

Hey there, 

Congrats to the winners and thanks a lot for congrats to my side :-)


Sharing details on my solution if anyone is interested, basically it is FFM model averaged for last 5 runs:

Technologies / frameworks:

1. Early merging of files was done using Python. Basically, I merged all files into  train/test/sumission files, then was reading line by line for generating features

2. All feature generation and statistics building is done using Scala. I found it way faster than Python + I am better with it, especially when it came to parallelization. 
I have also filtered page_views.csv to users seen in train/submission sets, so different statistics would fit in RAM. Size of FFM train file ~180G. RAM required ~ 80G

3. I have not used hashing for features, each feature seen >=10 times would get its own ID. Submission features unseen features in train are dropped

4. For numerical features, I have different strategies of dealing with them - bins, log smoothing or leaving as is, best chosen for each, based on validation. Some fields contain single feature, some bag of features. Total number of features is >5M

5. I added MEAP metric into libffm, so it would stop based on it, not on logloss. Used default parameters of libffm, including k=4

6. No special approach was taken to deal with leak, removing  it reduces model's precision + model learned almost all leaks naturally, so no post processing was required

7. Most of the features engineering was done on 32Gb laptop, later I used 128Gb RAM  machine and for the last week used AWS


8. What have not worked for me - GBT leaves as features, users clustering , MF of page_views.csv, stacking of FFM. Also, run out of time for stacking ffm->xgboost. 


9. List of features:

> country, state, platform, county, pageDocumentCategories,
> pageDocumentEntities, pageDocumentTopics, publisherCTRAdv,
> publisherCTRCamp, publisherCTRDoc, countryCTRAdv, countryCTRCamp,
> countryCTRDoc, stateCTRAdv, stateCTRCamp, stateCTRDoc, countyCTRAdv,
> countyCTRCamp, countyCTRDoc, prevDayCTRAdv, prevDayCTRCamp,
> prevDayCTRDoc, currentDayCTRAdv, currentDayCTRCamp, currentDayCTRDoc,
> nextDayCTRAdv, nextDayCTRCamp, nextDayCTRDoc, currentHourCTRAdv,
> currentHourCTRCamp, currentHourCTRDoc, prevHourCTRAdv,
> prevHourCTRCamp, prevHourCTRDoc, nextHourCTRAdv, nextHourCTRCamp,
> nextHourCTRDoc, publisherSourceCTRAdv, publisherSourceCTRCamp,
> publisherSourceCTRDoc, adId, documentId, campaignId, advertiserId,
> userPageViewsMeta1, userPageViewsMeta2, documentIdViews,
> campaignIdViews, advertiserIdViews, userDocsSeenFromLogToday,
> userPageMeta1SeenToday, userPageMeta2SeenToday,
> userCampSeenFromLogToday, userAdvertisersSeenFromLogToday,
> userDocsSeenFromLogTomorrow, userPageMeta1SeenTomorrow,
> userPageMeta2SeenTomorrow, docStats, documentCategoriesId, advStats,
> documentEntitiesId, documentTopicsId, metaSourceId, metaPublisherId,
> pageDocumentMeta1, pageDocumentMeta2, userClickedThisDocumentTimes,
> userSkippedThisDocumentTimes, userClickedThisCampaignTimes,
> userSkippedThisCampaignTimes, userClickedThisAdvertiserTimes,
> userSkippedThisAdvertiserTimes, hourOfDay, dayNum, dayOfWeek,
> thisAdEntityClickedBefore, daysFromAdDocPublished,
> thisAdEntityClickedBeforeCount, thisAdCategoryClickedBefore,
> thisAdCategoryClickedBeforeCount, userClickedThisAdTimes,
> userSkippedThisAdTimes, userPageViewsDocuments,
> userPageViewsCampaigns, userPageViewsAdvertisers,
> userPageViewsCategories, userPageViewsEntitites, userPageViewsTopics,
> seenThisCategoryInLog, userSeenThisMeta1, userSeenThisMeta2,
> seenThisDocInLog, seenThisCampaignInLog, seenThisAdvInLog, campStats,
> adStats, userCampSeenFromLogTomorrow,
> userAdvertisersSeenFromLogTomorrow, seenThisEntityInLog,
> userAllAdsFreq, nextDocUserClicked, nextCampUserClicked,
> nextAdvUserClicked, userClickedCount, userSkippedCount,
> userClickedCountToday, userDocsClickedToday, userCampClickedToday,
> userAdvertisersClickedToday, seenThisTopicInLog,
> userClickedThisDocFreq, userClickedThisCampFreq,
> userClickedThisAdvFreq, lastPageViewUserSeen, nextPageViewUserSeen,
> lastDocUserClicked, lastCampUserClicked, lastAdvUserClicked


Thanks,

Andrii
