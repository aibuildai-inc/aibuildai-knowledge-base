# 2nd position (solution)

Competition: mlsp-2014-mri
Rank: #2
Source: https://www.kaggle.com/c/mlsp-2014-mri/discussion/9854

<p>Hi Dear Colleagues,</p>
<p>I just wanted to share my solution that&nbsp;led me to the 2nd place:</p>
<p><a href="https://github.com/alex-lebedev/Kaggle-MLSP-2014">https://github.com/alex-lebedev/Kaggle-MLSP-2014</a></p>
<p>I had some problems during submissions. I was first receiving errors, although my csv-files looked just fine. I tried different formats, one of which has finally worked.</p>
<p>I implemented &quot;feature trimming&quot; in my solution, which consists of&nbsp;1) introducing a random vector into the feature set, 2) calculating feature importance, 3) removing the features with importance below the&nbsp;&quot;dummy feature&quot;.</p>
<p>Yes, it's as simple as that...</p>
<p>As I mentioned <a href="https://www.kaggle.com/c/mlsp-2014-mri/forums/t/9834/is-there-still-an-opportunity-to-submit-my-code">PREVIOUSLY</a>, I did not observe substantial progress and eventually gave up (after the third try). This is why I didn't&nbsp;even try more advanced techniques&nbsp;like boosting, and SCAD-SVM, Elastic Net, recursive feature elimination. Neither did I try building hierarchical models, which I had in mind, but did not implement for&nbsp;the same reasons.</p>
