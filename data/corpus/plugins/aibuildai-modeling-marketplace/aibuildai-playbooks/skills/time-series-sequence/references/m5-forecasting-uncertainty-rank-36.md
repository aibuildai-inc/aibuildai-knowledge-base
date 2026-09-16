# 36th Position Solution

Competition: m5-forecasting-uncertainty
Rank: #36
Source: https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163099

I would first like to thank @muhakabartay for his constant motivation to improve with every competition. Our solutions are based on 5 different models 

**Preprocessing**

We used preprocessing from the kernels below.

1. Ulrich GOUE's Kernel : https://www.kaggle.com/ulrich07/do-not-write-off-rnn-cnn
2. Ulrich GOUE's Kernel : https://www.kaggle.com/ulrich07/quantile-regression-with-keras  

**Modelling**

*Model 1 Architecture*

&gt; x = L.GRU(400, return_sequences=True, name="d1")(x)
x = L.Dropout(0.5)(x)
x = L.Concatenate(name="m1")([x, context])
x = L.GRU(400, return_sequences=True, name="d2")(x)
x = L.Dropout(0.5)(x)
x = L.Concatenate(name="m2")([x, context])
x = L.GRU(400, return_sequences=True, name="d3")(x)
preds = L.Dense(9, activation="linear", name="preds")(x)
model = M.Model(inp, preds, name="M1")
model.compile(loss=wqloss, optimizer="adam")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

*Model 2 Architecture*

&gt; x = L.GRU(410, return_sequences=True, name="d1")(x)
x = L.Dropout(0.5)(x)
x = L.Concatenate(name="m1")([x, context])
x = L.GRU(410, return_sequences=True, name="d2")(x)
x = L.Dropout(0.5)(x)
x = L.Concatenate(name="m2")([x, context])
x = L.GRU(410, return_sequences=True, name="d3")(x)
preds = L.Dense(9, activation="linear", name="preds")(x)
model = M.Model(inp, preds, name="M1")
model.compile(loss=wqloss, optimizer="adam")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)

*Model 3 Architecture*

&gt; x = L.Concatenate(name="x1")([context, num])
x = L.Dense(500, activation="relu", name="d1")(x)
x = L.Dropout(0.2)(x) # 0.3
x = L.Concatenate(name="m1")([x, context])
x = L.Dense(500, activation="relu", name="d2")(x)
x = L.Dropout(0.2)(x) # 0.3
x = L.Concatenate(name="m2")([x, context])
x = L.Dense(500, activation="relu", name="d3")(x)
preds = L.Dense(9, activation="linear", name="preds")(x)
model = M.Model(inp, preds, name="M1")
model.compile(loss=qloss, optimizer="adam")
return model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

*Model 4 Architecture*

&gt; x = L.Concatenate(name="x1")([context, num])
x = L.Dense(400, activation="relu", name="d1")(x)
x = L.Dropout(0.25)(x) # 0.3
x = L.Concatenate(name="m1")([x, context])
x = L.Dense(400, activation="relu", name="d2")(x)
x = L.Dropout(0.25)(x) # 0.3
x = L.Concatenate(name="m2")([x, context])
x = L.Dense(400, activation="relu", name="d3")(x)
preds = L.Dense(9, activation="linear", name="preds")(x)
model = M.Model(inp, preds, name="M1")
model.compile(loss=qloss, optimizer="adam")
return model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.01)

Model 5 : Baseline model https://www.kaggle.com/ulrich07/quantile-regression-with-keras  

Further we ensembled our 5 submissions. Here is our winning ensemble.

https://www.kaggle.com/roydatascience/silver-medal-solution-stacking-m5-submissions

Congratulations to all the winners. Happy Kaggling everyone.
