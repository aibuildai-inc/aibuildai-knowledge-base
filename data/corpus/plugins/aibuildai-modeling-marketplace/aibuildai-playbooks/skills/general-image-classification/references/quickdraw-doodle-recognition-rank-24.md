# 24th solution

Competition: quickdraw-doodle-recognition
Rank: #24
Source: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73701

You may not be interested, but our 24th solution is below.

-. data generate

-. Model

-. optimize, loss, etc

-. ensemble



1. data generate

 - Using simplified file



def draw_cv2_color(raw_strokes, size=256, lw=6, time_color=True):

    img = np.zeros((BASE_SIZE, BASE_SIZE,3), np.uint8)

    for t, stroke in enumerate(raw_strokes):     

        inertia_x = 0

        inertia_y = 0

        for i in range(len(stroke[0]) - 1):

            color = int(255 - 245*(float(t)/len(raw_strokes))) if time_color else 255 ## strokes order
            print(color)

            sx = stroke[0][i]

            sy = stroke[1][i]

            ex = stroke[0][i + 1]

            ey = stroke[1][i + 1]

            color_v =  (np.sqrt((sx-ex)*(sx-ex) + (sy-ey)*(sy-ey)) / np.sqrt(size*size)) * 255 ## strokes distance like speed (1~0)

            color_a = (np.sqrt((inertia_x-ex)*(inertia_x-ex) + (inertia_y-ey)*(inertia_y-ey)) / np.sqrt(size*size*4)) * 255 ## strokes distance like acceleration (1~0)

            _ = cv2.line(img, (sx, sy), (ex, ey), (color,color_v,color_a), lw)

            inertia_x = 2*ex -sx

            inertia_y = 2*ey-sy

            

    if size != BASE_SIZE:

        return cv2.resize(img, (size, size))

    else:

        return img





 - Using raw file





def draw_cv2_color_new(raw_strokes, size=256, lw=6, time_color=True, last_drop_r = 0.0):

    stx_min, sty_min = 99999, 99999

    stx_max, sty_max = 0,0

    ett=0  # How fast to complete less than 20 seconds

    

    for t, stroke in enumerate(raw_strokes):

        if t == len(raw_strokes) -1:

            ett = int(stroke[2][-1])

        for i in range(len(stroke[0])):

            stx_min = min(stx_min, int(stroke[0][i]))

            stx_max = max(stx_max, int(stroke[0][i]))

            sty_min = min(sty_min, int(stroke[1][i]))

            sty_max = max(sty_max, int(stroke[1][i]))

    

    limit_ett = 20*1000   

    ofs = 15

    

    if int(sty_max-sty_min+2*ofs) &gt; 6000 or int(stx_max-stx_min+2*ofs)  &gt; 6000:

        img = np.zeros((6000,6000,3), np.uint8)

    else:

        img = np.zeros((int(sty_max-sty_min+2*ofs), int(stx_max-stx_min+2*ofs),3), np.uint8)



    for t, stroke in enumerate(raw_strokes):     

        inertia_x = 0

        inertia_y = 0

        pre_st_t = 0 

        for i in range(len(stroke[0]) - 1):

            color = int(255 - 245*float(t)/len(raw_strokes)) if time_color else 255 ##  stroke order

            sx = int(stroke[0][i]) - stx_min +ofs

            sy = int(stroke[1][i]) - sty_min +ofs

            st = stroke[2][i]

            ex = int(stroke[0][i + 1])- stx_min +ofs

            ey = int(stroke[1][i + 1])- sty_min +ofs

            et = stroke[2][i+1]

            

            

            time = et-st

            if time ==0:

                time = 1

            

            color_v =  min(int((np.sqrt((sx-ex)*(sx-ex) + (sy-ey)*(sy-ey)) / time)*255.0), 255) ## speed

            color_a = min(int((np.sqrt((inertia_x-ex)*(inertia_x-ex) + (inertia_y-ey)*(inertia_y-ey)) / np.sqrt(time*time))*255.0), 255) ## acceleration (1~0)

            _ = cv2.line(img, (sx, sy), (ex, ey), (color,color_v,color_a), lw)

            

            if i==0:

                color_inter = int((float(et-pre_st_t)/limit_ett)*245)+10

                _ = cv2.circle(img, (sx, sy), lw, (0,0,color_inter), -1) ##interval time

            

            if i==len(stroke[0])-2 and t == len(raw_strokes) -1:

                color_end = int((float(ett)/(limit_ett)*245))+10



                _ = cv2.circle(img, (sx, sy), lw, (0,color_end,0), -1) ##end time

                

            inertia_x = 2*ex -sx

            inertia_y = 2*ey-sy

            pre_st_t=et

            

    return cv2.resize(img, (size, size)) #lw reflects how big the picture is drawn, also the aspect ratio is reflected









2.. Model Structure

- Best single model 

 : InceptionResnetV2 (139,139,3) size input  local valid score is  0.9516.

 : using raw file, using 'imagenet' weights, batch size 180





base_model = InceptionResNetV2(input_shape=input_shape, weights='imagenet',include_top= False)

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='rule')(x)

x = Dropout(0.3)(x)

predictions = Dense(340, activation='softmax', name='lastfc')(x)

model = Model(inputs=base_model.input, outputs=predictions)







3.. Optimizer, Loss, etc.

 - In first train about 1epoch(50M set) , I used adam and learning rate 0.002, 

   categorical cross entropy loss

 - Second train , I used adam accumulation 500 iters and learning rate 0.002, 

   categorical cross entropy 10% and top3 loss 90%.

 - I did not have enough time to train to the saturation.



#https://github.com/keras-team/keras/issues/3556



import keras.backend as K

from keras.legacy import interfaces

from keras.optimizers import Optimizer

class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):

        if accum_iters &lt; 1:

            raise ValueError('accum_iters must be &gt;= 1')

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))

        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay &gt; 0:

            lr = lr * (1. / (1. + self.decay * completed_updates))



        t = completed_updates + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))



        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)

        update_switch = K.cast(update_switch, K.floatx())



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]



        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):



            sum_grad = tg + g

            avg_grad = sum_grad / self.accum_iters_float



            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))

            else:

                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))

            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))

            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))

        return self.updates

    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))





4.. Ensemble

- I used weighed average ensemble by local valid score and argmax corr.

- I used InceptionResNetV2 (raw, simple), Xception (raw), Resnet50 (simple).

- I did not have time to train both (raw, simple)

- Average weight calculation is below code



def get_score_w(local_score):

    ls = np.array(local_score)

    sub = ls - ls.min()

    div = sub/sub.max()

    add = div + 0.1

    nor = add/add.max()

    sqr = nor*nor

    print(sqr)

    return sqr



def get_corr_w(clsnp):

    corxlist=[]

    for idx1, cls1 in enumerate((clsnp)):

        corylist=[]

        for idx2, cls2 in enumerate((clsnp)):

            cor_max = np.corrcoef(np.argmax(clsnp[idx1],axis=1),np.argmax(clsnp[idx2],axis=1))[0][1]

            corylist.append(cor_max)

        corxlist.append(corylist)



    df = pd.DataFrame(corxlist,columns=names, index=names)

    #print(df)

    corr_w = []

    for i in range(df.shape[0]):

        count = 0

        thr = 0.95

        for v in df.values[i]:

            if v &gt; thr :

                count+=1

        corr_w.append(1.0/count)   

    return np.array(corr_w), df



def getensemble_w(clsnp, local_score):

    score_w = get_score_w(local_score)

    for i in range(score_w.shape[0]):

        if score_w[i]==1.0:

            score_w[i] = 1.2 #max score add 20%

    corr_w = get_corr_w(clsnp)[0]

    ensemble_w = score_w*corr_w

    return ensemble_w





Everybody enjoy the competition~!!
