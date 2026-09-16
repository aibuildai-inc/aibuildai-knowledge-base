# Solution Rank 46th  with Some Feature corrections..

Competition: nbme-score-clinical-patient-notes
Rank: #45
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322839

. We used MLM trainings of the models  and below feature correction in one ensembled models.  PLS lead to some how overfitting of cv so we mixed it with Non PLs models to get gain and neutralize the overfit part of it.
Our Best final solution was mix of
1) Deb v3 Large
2) Deb   large
3) Deb v3 Large PL 

Second best selected solution was mix of  models
4 models ->deb l + debv3 No PL+ debv3 feature corrections No PL  +deberta base PL

Based on the error analysis or missing places in predictions  i came up with below feature corrections.
It did improve some fold wise CV dramatically but unfortunately LB dint see that much gain.
**If any one wants to try out their best solutions with these corrections try it out and let me know the results.** 

```
 print(features.iloc[132,2])
    features.iloc[132,2]='Vomiting-or-V'
    print(features.iloc[73,2])
    features.iloc[73,2]='Episodes-of-heart-racing-or-heart-poundings-or-palpitations'
    print(features.iloc[90,2])
    features.iloc[90,2]='Recent-upper-respiratory-infection-symptoms or-nasal-congestion'
    print(features.iloc[119,2])
    features.iloc[119,2]='Diminished-energy-or-feeling-drained-or-somnolence'
    print(features.iloc[76,2])
    features.iloc[76,2]='No-illicit-drug-use-or-No-toxic-habbits'
    print(features.iloc[6,2])
    features.iloc[6,2]='Adderall-use-or-Amphetamine'
    print(features.iloc[9,2])
    features.iloc[9,2]='heart-pounding-OR-heart-racing-or-palpitations'
    print(features.iloc[102,2])
    features.iloc[102,2]='LMP-2-months-ago-or-Last-menstrual-period-2-months-ago'
    print(features.iloc[[12,51,88],2])
    features.iloc[[12,51,88],2]='Male-or-M-or-Mr.'
    print(features.iloc[131,2]) 
    features.iloc[131,2]='Neck-pain-or-Pain-neck'
    print(features.iloc[135,2]) 
    features.iloc[135,2]='viral-symptoms-OR-rhinorrhea-OR-scratchy-throat(ST)'
    #new additions
    print(features.iloc[95,2]) 
    features.iloc[95,2]='No-shortness-of-breath-or-denies-dyspnea'
    print(features.iloc[63,2]) 
    features.iloc[63,2]='No-depressed-mood(good energy level)'
    print(features.iloc[2,2]) 
    features.iloc[2,2]='Chest-pressure-or-chest-pain'
    print(features.iloc[4,2]) 
    features.iloc[4,2]='Lightheaded-or-dizziness-or-presyncopal'
    #print(features.iloc[3,2]) 
    #features.iloc[3,2]='Intermittent-symptoms-or-random-episode'
    #print(features.iloc[74,2]) 
    #features.iloc[74,2]='Recent-visit-to-emergency-department(ED/ER)-with-negative-workup'
    print(features.iloc[97,2]) 
    features.iloc[97,2]='No-or-minor-relief-with-asthma-inhaler'
    print(features.iloc[82,2]) 
    features.iloc[82,2]='Feels-hot-OR-Feels-clammy-or-sweat'
    print(features.iloc[77,2]) 
    features.iloc[77,2]='Associated-nausea(N)'
    print(features.iloc[81,2]) 
    features.iloc[81,2]='Associated-throat-tightness-or-throat-narrowing'
    print(features.iloc[54,2]) 
    features.iloc[54,2]='No-blood-in-stool-or-BRBPR'
    print(features.iloc[129,2]) 
    features.iloc[129,2]='Myalgias-or-muscle-achiness'
    #print(features.iloc[50,2]) 
    #features.iloc[50,2]='2-to-3-beers-a-week-or-quit-or-few'
    print(features.iloc[142,2]) 
    features.iloc[142,2]='Subjective-fever-or-high-temperature'
    print(features.iloc[117,2]) 
    features.iloc[117,2]='Unsuccessful-napping-or-no-naps'
    print(features.iloc[[57,134],2]) 
    features.iloc[[57,134],2]='Nausea-or-N'
    #features.iloc[5,2]='No-hair-changes-OR-no-nail-changes-OR-no-temperature-intolerance-or-no-
```
