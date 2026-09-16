# 11th place brief solution

Competition: g2net-gravitational-wave-detection
Rank: #11
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275405

hi, im a student learning Computer science/Statistics at university and im new to Kaggle.✋
I don't really know much about GW/signal processing/ or other deep learning skills(augmentation, stacking,etc) so I just mainly stuck to modeling(1D CNNs).

**preprocessing:**
1) bpf 30-500
2) bpf 25-1020

**augmentation:**
mixup

**models:**
single 1DCNN model v1 -> cv 8800 lb 8800
single 1DCNN model v2 -> cv 877x lb ?(didn't try)

v1 + v2 average ensemble -> cv 882x public lb 882x (!)
v1 + v2 with preprocessing1,2 -> cv 883x public lb 8830
using whole dataset -> public lb8833
stacking with a single cqt model in public notebook(cv875x, thanks to @ragnar123)
-> public lb8836

for 1DCNN models, I used residual connection and dilated convolution/standard convolution with ~24 layers, ~2M parameters.
I tried to figure out why v1+v2 ensembling gets so much improvement and tried to build a model with both models' advantages but I couldn't due to lack of time.

so glad for learning so many things/achieving a gold medal in my very first DL competition!
And special thanks to @hidehisaarai1213 and @miklgr500 for providing such a great notebook to start with:)
