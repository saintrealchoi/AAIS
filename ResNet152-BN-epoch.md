```python
!nvidia-smi
```

    Thu Feb  4 11:54:44 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.104      Driver Version: 410.104      CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0 Off |                    0 |
    | N/A   57C    P0   190W / 300W |   8653MiB / 32478MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   54C    P0   148W / 300W |  12391MiB / 32478MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   50C    P0    54W / 300W |   1792MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   50C    P0    52W / 300W |   7986MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+



```python
# using gpu:/0
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)
```

**test용**


0-19 = 100명


20-49 = 75명


50- = 51명

= 226 * 4 = 804개의 image

**train용**


0-19 = 299명


20-49 = 225명


50- = 154명

= 678 * 4 = 2712개의 image


```python
path_dir1 = '../sungjin/tooth3/train/'
path_dir2 = '../sungjin/tooth3/test/'

sub_directory = ['0/','20/','50/']

TRAIN_NUM = 0
TEST_NUM = 0

for subdir in sub_directory:
    tmp_list = os.listdir(path_dir1+subdir)
    TRAIN_NUM += len(tmp_list)
    
for subdir in sub_directory:
    tmp2_list = os.listdir(path_dir2+subdir)
    TEST_NUM += len(tmp2_list)
    
```


```python
print(TRAIN_NUM)
```

    2812



```python
print(TEST_NUM)
```

    807



```python
import numpy as np

# preprocessing image

train_img = np.float32(np.zeros((TRAIN_NUM,224,224,3)))
train_label = np.float64(np.zeros((TRAIN_NUM,7)))

test_img = np.float32(np.zeros((TEST_NUM,224,224,3)))
test_label = np.float64(np.zeros((TEST_NUM,7)))
```

## Make Label

ImageDataGenerator를 사용하지 않을때 labelling을 일일이 함

그러나 data augmentation을 이용하기 위해 폴더별로 사용


```python
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.resnet50 import preprocess_input

# make_label = [0,10,20,30,40,50,60]

# num = 0
# idx = 0
# for subdir in sub_directory:
#     give_label = make_label[idx]
#     idx += 1
#     for file in os.listdir(path_dir1+subdir):
#         src_path = path_dir1 + subdir + file
#         img = load_img(src_path, target_size = (224,224))

#         x = img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         train_img[num, :, :, :] = x

#         train_label[num][idx-1] = give_label
#         num = num + 1
```


```python

# num = 0
# idx = 0

# for subdir in sub_directory:
#     give_label = make_label[idx]
#     idx += 1
#     for file in os.listdir(path_dir2+subdir):
#         src_path = path_dir2 + subdir + file
#         img = load_img(src_path, target_size = (224,224))

#         x = img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         test_img[num, :, :, :] = x

#         test_label[num][idx-1] = give_label
#         num = num + 1
```

## Data Augmentation


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        horizontal_flip=True,  # randomly flip images left and right
        vertical_flip=True)  # randomly flip images upside down
```


```python
# load and iterate training dataset

seed = 1
    
train_it = datagen.flow_from_directory('../sungjin/tooth3/train',
                                       seed = seed,
                                       target_size=(224, 224), 
                                       class_mode='categorical', 
                                       batch_size=32)
# load and iterate test dataset
test_it = datagen.flow_from_directory('../sungjin/tooth3/test',
                                      seed = seed,
                                      target_size=(224, 224),
                                      class_mode='categorical', 
                                      batch_size=32)
```

    Found 2812 images belonging to 3 classes.
    Found 808 images belonging to 3 classes.


## Transfer Learning


```python
import tensorflow as tf
from keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,AveragePooling2D,Flatten,Dropout
from tensorflow import keras

base_model = keras.applications.ResNet152(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)
```


```python
base_model.trainable = False
```

### Classification 층만 삭제하고 FC-BN


```python
inputs = tf.keras.Input(shape = (224,224,3))

x = base_model(inputs, training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(3,activation = 'softmax')(x)
model = tf.keras.Model(inputs,outputs)
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    resnet152 (Functional)       (None, 7, 7, 2048)        58370944  
    _________________________________________________________________
    flatten (Flatten)            (None, 100352)            0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              102761472 
    _________________________________________________________________
    batch_normalization (BatchNo (None, 1024)              4096      
    _________________________________________________________________
    dense_1 (Dense)              (None, 3)                 3075      
    =================================================================
    Total params: 161,139,587
    Trainable params: 102,766,595
    Non-trainable params: 58,372,992
    _________________________________________________________________



```python
model.compile(optimizer = keras.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
```


```python
history = model.fit(train_it, steps_per_epoch=20, validation_data=test_it, validation_steps=10, epochs=10)
```

    Epoch 1/10
    20/20 [==============================] - 151s 7s/step - loss: 3.8793 - accuracy: 0.4900 - val_loss: 5.0019 - val_accuracy: 0.6406
    Epoch 2/10
    20/20 [==============================] - 95s 5s/step - loss: 1.1653 - accuracy: 0.6475 - val_loss: 2.3775 - val_accuracy: 0.6313
    Epoch 3/10
    20/20 [==============================] - 93s 5s/step - loss: 0.8492 - accuracy: 0.6679 - val_loss: 1.8928 - val_accuracy: 0.6313
    Epoch 4/10
    20/20 [==============================] - 93s 5s/step - loss: 0.7203 - accuracy: 0.7301 - val_loss: 1.2352 - val_accuracy: 0.6687
    Epoch 5/10
    20/20 [==============================] - 94s 5s/step - loss: 0.6541 - accuracy: 0.7297 - val_loss: 1.1352 - val_accuracy: 0.6687
    Epoch 6/10
    20/20 [==============================] - 92s 5s/step - loss: 0.5476 - accuracy: 0.7729 - val_loss: 1.1142 - val_accuracy: 0.6938
    Epoch 7/10
    20/20 [==============================] - 94s 5s/step - loss: 0.6131 - accuracy: 0.7372 - val_loss: 0.8507 - val_accuracy: 0.6687
    Epoch 8/10
    20/20 [==============================] - 92s 5s/step - loss: 0.5988 - accuracy: 0.7358 - val_loss: 0.9383 - val_accuracy: 0.6687
    Epoch 9/10
    20/20 [==============================] - 92s 5s/step - loss: 0.5959 - accuracy: 0.7362 - val_loss: 0.8256 - val_accuracy: 0.6969
    Epoch 10/10
    20/20 [==============================] - 92s 5s/step - loss: 0.5435 - accuracy: 0.7549 - val_loss: 0.8801 - val_accuracy: 0.6750


history = model.fit(train_img, train_label, epochs=20, batch_size=32, validation_data=(test_img, test_label))


```python
import numpy as np
import matplotlib.pyplot as plt

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))


plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![png](output_21_0.png)
    



```python
y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

x_len = np.arange(len(y_acc))

plt.plot(x_len, y_vacc, marker='.', c='red')
plt.plot(x_len, y_acc, marker='.', c='blue')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

    No handles with labels found to put in legend.



    
![png](output_22_1.png)
    


## Unfreeze and learning base model with low learning rate


```python
base_model.trainable = True

model.compile(optimizer = keras.optimizers.Adam(lr = .00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```python
history = model.fit(train_it, steps_per_epoch=20, validation_data=test_it, validation_steps=10, epochs=10)
```

    Epoch 1/10
    20/20 [==============================] - 200s 9s/step - loss: 0.6037 - accuracy: 0.7390 - val_loss: 0.8981 - val_accuracy: 0.7063
    Epoch 2/10
    20/20 [==============================] - 177s 9s/step - loss: 0.5288 - accuracy: 0.7641 - val_loss: 0.7890 - val_accuracy: 0.6938
    Epoch 3/10
    20/20 [==============================] - 174s 9s/step - loss: 0.5356 - accuracy: 0.7349 - val_loss: 0.7289 - val_accuracy: 0.7125
    Epoch 4/10
    20/20 [==============================] - 176s 9s/step - loss: 0.4807 - accuracy: 0.7707 - val_loss: 0.6688 - val_accuracy: 0.7188
    Epoch 5/10
    20/20 [==============================] - 182s 9s/step - loss: 0.4892 - accuracy: 0.7972 - val_loss: 0.6457 - val_accuracy: 0.7281
    Epoch 6/10
    20/20 [==============================] - 230s 12s/step - loss: 0.5083 - accuracy: 0.7609 - val_loss: 0.6362 - val_accuracy: 0.7281
    Epoch 7/10
    20/20 [==============================] - 244s 12s/step - loss: 0.4983 - accuracy: 0.7540 - val_loss: 0.7737 - val_accuracy: 0.7000
    Epoch 8/10
    20/20 [==============================] - 221s 11s/step - loss: 0.5064 - accuracy: 0.8042 - val_loss: 0.8217 - val_accuracy: 0.6719
    Epoch 9/10
    20/20 [==============================] - 176s 9s/step - loss: 0.4589 - accuracy: 0.7891 - val_loss: 0.7423 - val_accuracy: 0.7156
    Epoch 10/10
    20/20 [==============================] - 173s 9s/step - loss: 0.4325 - accuracy: 0.8271 - val_loss: 0.7001 - val_accuracy: 0.7281



```python
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))


plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![png](output_26_0.png)
    



```python
y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

x_len = np.arange(len(y_acc))

plt.plot(x_len, y_vacc, marker='.', c='red')
plt.plot(x_len, y_acc, marker='.', c='blue')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```


    
![png](output_27_0.png)
    



```python
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)

def make_predictions(image_path):
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    
    return np.argmax(preds)
```


```python
make_predictions('../sungjin/tooth3/test/20/20683928 김영중 16.png')
```




    1




    
![png](output_29_1.png)
    



```python
make_predictions('../sungjin/tooth3/test/20/20683928 김영중 26.png')
```




    1




    
![png](output_30_1.png)
    



```python
make_predictions('../sungjin/tooth3/test/20/20683928 김영중 36.png')
```




    1




    
![png](output_31_1.png)
    



```python
make_predictions('../sungjin/tooth3/test/20/20683928 김영중 46.png')
```




    0




    
![png](output_32_1.png)
    



```python
def predict_agegroup(image_path):
    preds = make_predictions(image_path)
    if preds == 0:
        print("나이는 0~19세 사이일 것입니다.")
    elif preds == 1:
        print("나이는 20~49세 사이일 것입니다.")
    else:
        print("나이는 50세 이상일 것입니다.")
```


```python
predict_agegroup('../sungjin/tooth3/test/20/20683982 기선미 16.png')
```

    나이는 20~49세 사이일 것입니다.



    
![png](output_34_1.png)
    



```python
predict_agegroup('../sungjin/tooth3/test/20/20683982 기선미 26.png')
```

    나이는 20~49세 사이일 것입니다.



    
![png](output_35_1.png)
    



```python
predict_agegroup('../sungjin/tooth3/test/20/20683982 기선미 36.png')
```

    나이는 20~49세 사이일 것입니다.



    
![png](output_36_1.png)
    



```python
predict_agegroup('../sungjin/tooth3/test/20/20683982 기선미 46.png')
```

    나이는 50세 이상일 것입니다.



    
![png](output_37_1.png)
    



```python

```


```python

```


```python
import pandas as pd

p = model.predict_generator(test_it, verbose=True)
pre = pd.DataFrame(p)
```

    /root/anaconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1905: UserWarning: `Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.
      warnings.warn('`Model.predict_generator` is deprecated and '


    26/26 [==============================] - 58s 2s/step



```python
pre["filename"] = test_it.filenames
```


```python
pre["label"] = (pre["filename"].str.contains("0")).apply(int)
```


```python
pre['pre'] = (pre[1]>0.5).apply(int)
```


```python
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score

recall_score(pre["label"],pre["pre"])
```




    0.4291044776119403




```python
roc_auc_score(pre["label"],pre[1])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-39-a8ee5505d471> in <module>
    ----> 1 roc_auc_score(pre["label"],pre[1])
    

    ~/anaconda3/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 


    ~/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_ranking.py in roc_auc_score(y_true, y_score, average, sample_weight, max_fpr, multi_class, labels)
        388         labels = np.unique(y_true)
        389         y_true = label_binarize(y_true, classes=labels)[:, 0]
    --> 390         return _average_binary_score(partial(_binary_roc_auc_score,
        391                                              max_fpr=max_fpr),
        392                                      y_true, y_score, average,


    ~/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_base.py in _average_binary_score(binary_metric, y_true, y_score, average, sample_weight)
         75 
         76     if y_type == "binary":
    ---> 77         return binary_metric(y_true, y_score, sample_weight=sample_weight)
         78 
         79     check_consistent_length(y_true, y_score, sample_weight)


    ~/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_ranking.py in _binary_roc_auc_score(y_true, y_score, sample_weight, max_fpr)
        221     """Binary roc auc score"""
        222     if len(np.unique(y_true)) != 2:
    --> 223         raise ValueError("Only one class present in y_true. ROC AUC score "
        224                          "is not defined in that case.")
        225 


    ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.



```python

```


```python
p = model.predict_generator(test_batches, verbose=True)
pre = pd.DataFrame(p)
pre["filename"] = test_batches.filenames
pre["label"] = (pre["filename"].str.contains("PNEUMONIA")).apply(int)
pre['pre'] = (pre[1]>0.5).apply(int)

```


```python
recall_score(pre["label"],pre["pre"])
```


```python
roc_auc_score(pre["label"],pre[1])

```


```python
tpr,fpr,thres = roc_curve(pre["label"],pre[1])
roc = pd.DataFrame([tpr,fpr]).T
roc.plot(x=0,y=1)
```


```python

```


```python

```


```python

```


```python
import numpy as np
import matplotlib.pyplot as plt

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))


plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![png](output_54_0.png)
    



```python
y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

x_len = np.arange(len(y_acc))

plt.plot(x_len, y_vacc, marker='.', c='red')
plt.plot(x_len, y_acc, marker='.', c='blue')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

    No handles with labels found to put in legend.



    
![png](output_55_1.png)
    



```python
base_model.trainable = True

model.compile(optimizer = keras.optimizers.Adam(lr = .00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```python
history = model.fit(train_it, steps_per_epoch=20, validation_data=test_it, validation_steps=10, epochs=10)
```

    Epoch 1/10
    20/20 [==============================] - 44s 2s/step - loss: 8.6131 - accuracy: 0.2297 - val_loss: 9.0161 - val_accuracy: 0.2281
    Epoch 2/10
    20/20 [==============================] - 43s 2s/step - loss: 9.0161 - accuracy: 0.2313 - val_loss: 8.9153 - val_accuracy: 0.2281
    Epoch 3/10
    20/20 [==============================] - 43s 2s/step - loss: 8.5879 - accuracy: 0.2031 - val_loss: 8.4116 - val_accuracy: 0.2156
    Epoch 4/10
    20/20 [==============================] - 43s 2s/step - loss: 8.9938 - accuracy: 0.2398 - val_loss: 9.1672 - val_accuracy: 0.2500
    Epoch 5/10
    20/20 [==============================] - 43s 2s/step - loss: 8.5879 - accuracy: 0.2203 - val_loss: 8.5627 - val_accuracy: 0.2281
    Epoch 6/10
    20/20 [==============================] - 43s 2s/step - loss: 8.7894 - accuracy: 0.2250 - val_loss: 9.0161 - val_accuracy: 0.2313
    Epoch 7/10
    20/20 [==============================] - 43s 2s/step - loss: 9.1420 - accuracy: 0.2266 - val_loss: 8.9153 - val_accuracy: 0.2438
    Epoch 8/10
    20/20 [==============================] - 43s 2s/step - loss: 8.8398 - accuracy: 0.2188 - val_loss: 8.7642 - val_accuracy: 0.2188
    Epoch 9/10
    20/20 [==============================] - 43s 2s/step - loss: 9.1924 - accuracy: 0.2188 - val_loss: 8.7138 - val_accuracy: 0.2281
    Epoch 10/10
    20/20 [==============================] - 42s 2s/step - loss: 8.8422 - accuracy: 0.2100 - val_loss: 8.9657 - val_accuracy: 0.2188



```python

```


```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}




```python

```
