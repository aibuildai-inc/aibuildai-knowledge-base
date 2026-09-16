# 7th place solution

Competition: pku-autonomous-driving
Rank: #7
Source: https://www.kaggle.com/c/pku-autonomous-driving/discussion/127034

Congrats everyone with excellent result!
I summarize and write down the part of my solution and our post process.
For other part:
[camaro](https://www.kaggle.com/bamps53) part: [(part of) 7th place solution with code](https://www.kaggle.com/c/pku-autonomous-driving/discussion/127056)
[Jhui He](https://www.kaggle.com/hesene) and [yelan](https://www.kaggle.com/lanjunyelan) part: [https://www.kaggle.com/c/pku-autonomous-driving/discussion/127034#726362](https://www.kaggle.com/c/pku-autonomous-driving/discussion/127034#726362)


.jpg?generation=1579652111438826&amp;alt=media)

## Model1: detection and pose estimation
### common setting
<strong>detection model</strong>
- Mask RCNN(mask head removed)
- backbone: resnext101-32x4d
- lvis pretrained

<strong>pose estimation model</strong>
- HRNet-w18c, efficientnetb0/b3
- imagenet pretrained

<strong>Loss</strong>
- classification: BCE
- detectoin: Focal Loss
- pose regression: L1 Loss

<strong>detection: Optimizer and scheduler</strong>
- optimizer: SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
- scheduler: CosineAnnealingWarmRestarts

<strong>pose estimation: Optimizer and scheduler</strong>
- optimizer: Adam(lr=0.0001)
- scheduler: None

<strong>Augmentation</strong>
- detection: horizontal flip
- pose estimation: horizontal flip, shift, rotate, random blightness/contrast

### 1. pretrain on the boxy-vehicle-dataset
At first, I train my model on the [boxy-vehicle-dataset](https://boxy-dataset.com/boxy/).
This dataset include axis aligned bounding box and 3d cuboids, but I use only 2d bbox.<br>
<strong>training setting</strong>:
- image resolution: 1232x1028
- epochs: 10
- batch_size: 4
### 2. finetune on competition dataset
<strong>Model</strong>
- add depth head on top of model

<strong>preprocess</strong>
- split train vs val = 9 vs 1
- create 3d bbox using label, then create axis aligned bbox
- depth -&gt; 1 / sigmoid(depth) - 1

<strong>training setting</strong>
- image resolution: 800 x 2800, 1400x3300
- depth loss: L1 Loss
- epochs: 50
- batch_size: 4

### 3. pose estimation(yaw_sin, yaw_cos, pitch)
<strong>preprocess</strong>
- crop image by bbox and resize

<strong>training setting</strong>
- image resolution: 320x480
- epochs: 30
- batch_size: 128

public LB/private LB
- 800x2800, single fold: 0.119/0.106
- 1400x3300, single fold: 0.106/0.111

# Model2: centernet
<strong>model</strong>
- [pytorch dla centernet](https://github.com/xingyizhou/CenterNet).
- regression of yaw_sin, yaw_cos, pitch, depth, 2d bbox size(w, h), 3d bbox size(w, h, l)
- classification of object centerness(heat map)

<strong>Loss</strong>
- regression: L1 Loss
- classification: Focal Loss

<strong>optimizer and scheduler</strong>
- optimizer: Adam(lr=5e-4)
- scheduler: CosineAnnealingLR(lr=5e-5)

<strong>Augmentation</strong>
- pose estimation: horizontal flip, shift, random blightness/contrast

<strong>preprocess</strong>
- depth -&gt; 1 / sigmoid(depth) - 1

<strong>training setting</strong>
- split train vs val = 8 vs 2
- epochs: 30
- batch_size: 12

public LB/ private LB
- single fold: 0.100/0.096

# Ensemble: Linear assignment
We use different model(faster rcnn, centernet), so it is difficult to ensemble predicitons.
So we decided to ensemble nearset points between predictions.
We use hungalian algorithm for linear assignment.
Please refere to below code.
```
from scipy.optimize import linear_sum_assignment
distance_th = 30
yaw_th = 10

sub1 = pd.read_csv('sub1.csv')
y1 = sub1['PredictionString'].str.split(' ').values
X1 = sub1['ImageId'].values

sub2 = pd.read_csv('sub2.csv')
y2 = sub2['PredictionString'].str.split(' ').values
X2 = sub2['ImageId'].values
​
for idx in tqdm(range(len(sub1)), position=0):
    if str(np.nan) != str(y1[idx]) and str(np.nan) != str(y2[idx]):
        label1 = np.array(y1[idx]).reshape(-1, 7).astype(float)
        label2 = np.array(y2[idx]).reshape(-1, 7).astype(float)
        center_points1 = get_imgcoords(label1) # [N, 3], (img_x, img_y, img_z)
        center_points2 = get_imgcoords(label2)
        
        cost_matrix = np.zeros([len(center_points1), len(center_points1)])
        for idx1, i in enumerate(center_points1):
            for idx2, j in enumerate(center_points2):
                cost_matrix[idx1, idx2] = np.linalg.norm(i - j)
        
        match1, match2 = linear_sum_assignment(cost_matrix)
        for i, j in zip(match1, match2):
            if cost_matrix[i, j] &lt; distance_th:
                if np.abs(label1[i][1] - label2[j][1]) &lt; yaw_th:
                    label1[j] = (label1[i] + label2[j]) / 2
        label1[idx] = np.concatenate(tmp1).astype(str)

# use label1 for submission
```
