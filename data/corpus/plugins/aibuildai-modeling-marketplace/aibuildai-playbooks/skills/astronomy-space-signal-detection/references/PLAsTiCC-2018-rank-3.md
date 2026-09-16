# 3rd Place Part III - Pesudo Astronomer's Feature Engineering

Competition: PLAsTiCC-2018
Rank: #3
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75222

First of all, thanks to the organizers and all participants in this competition! And thanks a lot to my great teammates, @mamasinkgs, and @yuval6967. I learn a lot from these 2 guys. And I'm very happy to get my first gold medal :)

In this part, I want to share my findings and feature engineering. Please refer to the following discussions to see the overall description of our solution.

- [3rd Place Part I - CNN][1]
- [3rd Place Part II - CatBoost, mamas feature, and class 99][2]

## What I learned from astronomer's work
At the beginning of this competition, I tried to install domain knowledge and tried to be "pseudo astronomer". In addition to the data note provided by the organizers, I carefully read the following resources.

- [LSST science book][3]
- [Result from SNPCC challenge][4]
- [Photometric Supernova Classification With Machine Learning][5] ([slide][6])

After reading these resources and exploring data a bit, I thought that this competition mainly consists of these 2 challenges:

- How to distinguish between various types of supernova classes?
- How to detect class99?

I also tried to understand what each class really is. By 1) basic light curve characteristics 2) class frequency compared with expected LSST observation rate 3) redshift distribution, here is my assumption (order by confidence):

- class90: SN Ia
- class42: SN II
- class95: Superluminous Supernova 
- class52/62: SN Ib/c
- class88: Strong Gravitational Lens (or AGN?)
- class67: TDE

I hope the competition hosts will reveal the answer!
*I don't think this assumption helped us directly*, but this gave us a very good interpretation of the result of our class99 probing (class99 is something similar to core-collapse supernova).

## Feature Engineering
### template fitting by sncosmo package - the golden feature for us
sncosmo provides nice lc fitting API in python ([link](https://sncosmo.readthedocs.io/en/v1.6.x/examples/plot_lc_fit.html#sphx-glr-examples-plot-lc-fit-py)). I used a fitting parameter and it's chi-square value as a feature. I'm a bit surprised that no one except me uses this library because using template fitting is a major solution in the past competition.

It took a bit long time to calculate (1~2 lines/sec), so I split light curves by object_id%30 and run 30 preemptible instances to finish calculating within 1 day. I repeated this process about 15 times to make various types of models to cover all variant of supernovae (salt-2, salt-2-extended, nugent-sn1bc, nugent-sn2, snana, sako,...), and use 8 of them in my final model. It was an exhausting process... really...

But these bag of template features gave me a significant boost (~0.05 in intermediate, and 0.1+ in my final model). It helped Yuval's CNN as well (+0.04). 7 out of 10 my most important features (by LGBM gain) are these features. All template features were used only in an extragalactic model.

[example of template fitting result]

Here is an example of a result of template fitting (SALT-2, object_id = 161521). y-band is ignored by its wavelength and estimated redshift. One may think that GP fitting gives better fitting, but it's robust to noise (in SALT-2, only 5 parameters used to generate this all 6-band curve). Probably combining our features with GP features of Kyle or CPMP's will give us another significant boost.

#### *UPDATED*
I've attached my template features. you can download, unzip and use:

```
df = pd.read_feather('sncosmo_template_features.f')
```

#### *UPDATED2*
Kernel is available to see how did we use sncosmo:
https://www.kaggle.com/nyanpn/salt-2-feature-part-of-3rd-place-solution


### hostgal-specz model
Exactly same as 2nd and 4th place solution. This gave me a small boost (~0.004).


### luminosity
As shared in the discussion, luminosity is an important intrinsic property of variable stars. I used the following fomla:

```
luminosity = (max(flux) - min(flux)) * distance ** 2
```

Distance (in MPc) is converted from estimated specz above. Astropy's luminosity_distance ([link](http://docs.astropy.org/en/stable/cosmology/index.html?highlight=luminosity#using-astropy-cosmology)) function gives distance from redshift.

I added these luminosity and luminosity difference between channels.

### time difference
I made some features related to time-to-time difference. for example:

- (max(mjd) where detected == 1) - (mjd on peak flux)
- (mjd on peak flux) - (min(mjd) where detected == 1)
- (min(mjd) where detected == 1) - (mjd on previous observation)
    - tried to capture information where the important signal is lost by its observation schedule
- (mjd on 50% tile after the peak flux) - (mjd no peak flux)

### LombScargle
Adding power and frequency obtained from `astropy.LombScargle.autopower()` gave a small improvement.

## Pseudo Labelling
I also tried to use pseudo-labeling in the early stage of the competition. I found that using pseudo-label only in class90 gave me a big boost (0.005 ~ 0.03, depends on the model), but using all classes didn't work. I think it's related class99 (if class99 is similar to class52/62, pseudo-label in these classes contains a lot of false signals). class90+class42 gave a good result too, but its difference from class90-only was very small. 


  [1]: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75116
  [2]: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75131
  [3]: https://www.lsst.org/scientists/scibook
  [4]: https://arxiv.org/abs/1008.1024
  [5]: https://arxiv.org/abs/1603.00882
  [6]: https://kicp-workshops.uchicago.edu/SNClassification_2016/depot/talk-lochner-michelle.pdf
