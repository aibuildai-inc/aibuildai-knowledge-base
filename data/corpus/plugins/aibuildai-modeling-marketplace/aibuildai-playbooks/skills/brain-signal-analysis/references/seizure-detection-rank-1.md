# Required model documentation and code

Competition: seizure-detection
Rank: #1
Source: https://www.kaggle.com/c/seizure-detection/discussion/10111

<p>Hey everyone,</p>
<p>My code and documentation are now ready. It was&nbsp;great fun competing with you all,&nbsp;that final week was pretty intense!</p>
<p>https://github.com/MichaelHills/seizure-detection/raw/master/seizure-detection.pdf</p>
<p>https://github.com/MichaelHills/seizure-detection</p>
<p>Quickly summarising my model, for feature selection I used FFT 1-47Hz, concatenated with correlation coefficients (and their eigenvalues) of both the FFT output data, as well as the input time data. The data was then trained on per-patient Random Forest classifiers (3000 trees).</p>
