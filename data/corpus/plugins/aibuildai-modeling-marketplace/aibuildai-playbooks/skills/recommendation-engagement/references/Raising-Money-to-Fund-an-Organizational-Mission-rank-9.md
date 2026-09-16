# Post-Mortem

Competition: Raising-Money-to-Fund-an-Organizational-Mission
Rank: #9
Source: https://www.kaggle.com/c/Raising-Money-to-Fund-an-Organizational-Mission/discussion/2712

<p>Anybody do or find anything interesting in this dataset? Or find any good tools for working with large data sets?</p>
<p>I found it barely manageable given the size.&nbsp; It also took me awhile to wrap by head around what was in each of the files. Also the database restrictions (1 and 2 but not 3, 1 and 2 and 3, 2 and 3 but not 1, etc) made a really difficult task much harder
 and considerably less enjoyable.&nbsp; I ended up having to write 12 large files (and accompanying SQL code) for each model to compensate for this.
</p>
<p>I started out with a ~2% sample of the training data in R but even this was rough.&nbsp; I tried using the
<em>ff</em> package without much luck.&nbsp; I ended up doing most of the data manipulation using SQL Server 2008 R2 Express and the SSMS which I found to be a bright spot in the whole process as it performed really well given the data size.&nbsp; I especially appreciated
 the data import manager which helps with wide data sets.</p>
<p>My best model ended up just being prior averages for prospectid, zip5, and packageid with linear regression.&nbsp; I predicted donation amount (not amount2) and response rate using separate models.&nbsp; I'd then do predictedGift^1.15 * predictedResponseRate for a
 final prediction.</p>
<p>I tried to use some of the demographic data but had a hard time as I was using zip5 as the key to get state abbreviations as a factor, but some of the zip codes cross state lines which leads to duplicates and zip9, even when indexed, just took too long.</p>
<p>I think the contest was a cool idea but would have been much better on just one of the 3 databases.&nbsp; Without that restriction I would have had more time to explore the demographic and historical data.&nbsp; The one thing I would have really liked to explore is
 people's giving before the training period especially as much of the mailings seemed to be political and the test data is sitting within 12 months of a presidential election.
</p>
