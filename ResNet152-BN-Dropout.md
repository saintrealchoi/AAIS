```python
!nvidia-smi
```

    Thu Feb  4 11:44:19 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.104      Driver Version: 410.104      CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0 Off |                    0 |
    | N/A   57C    P0   158W / 300W |   8653MiB / 32478MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   54C    P0   225W / 300W |  12391MiB / 32478MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   49C    P0    54W / 300W |   1792MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   49C    P0    52W / 300W |   1523MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+



```python
# using gpu:/3
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

### Classification 층만 삭제하고 FC-BN-Dropout


```python
inputs = tf.keras.Input(shape = (224,224,3))

x = base_model(inputs, training=False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation = 'relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(3, activation = 'softmax')(x)
model = keras.Model(inputs,outputs)
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
    dropout (Dropout)            (None, 1024)              0         
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
history = model.fit(train_it, steps_per_epoch=20, validation_data=test_it, validation_steps=10, epochs=20)
```

    Epoch 1/20
    20/20 [==============================] - 88s 4s/step - loss: 2.9841 - accuracy: 0.5379 - val_loss: 8.2580 - val_accuracy: 0.6187
    Epoch 2/20
    20/20 [==============================] - 69s 3s/step - loss: 1.1867 - accuracy: 0.6760 - val_loss: 3.1783 - val_accuracy: 0.6187
    Epoch 3/20
    20/20 [==============================] - 69s 3s/step - loss: 0.8629 - accuracy: 0.7012 - val_loss: 2.1510 - val_accuracy: 0.6187
    Epoch 4/20
    20/20 [==============================] - 90s 5s/step - loss: 0.6907 - accuracy: 0.7229 - val_loss: 1.4343 - val_accuracy: 0.6062
    Epoch 5/20
    20/20 [==============================] - 96s 5s/step - loss: 0.7328 - accuracy: 0.7235 - val_loss: 1.5396 - val_accuracy: 0.6125
    Epoch 6/20
    20/20 [==============================] - 115s 6s/step - loss: 0.7231 - accuracy: 0.7090 - val_loss: 1.1240 - val_accuracy: 0.6656
    Epoch 7/20
    20/20 [==============================] - 101s 5s/step - loss: 0.5427 - accuracy: 0.7640 - val_loss: 0.9514 - val_accuracy: 0.6281
    Epoch 8/20
    20/20 [==============================] - 93s 5s/step - loss: 0.5612 - accuracy: 0.7609 - val_loss: 1.0714 - val_accuracy: 0.6719
    Epoch 9/20
    20/20 [==============================] - 94s 5s/step - loss: 0.5844 - accuracy: 0.7452 - val_loss: 1.2454 - val_accuracy: 0.6406
    Epoch 10/20
    20/20 [==============================] - 92s 5s/step - loss: 0.6004 - accuracy: 0.7379 - val_loss: 1.1194 - val_accuracy: 0.6500
    Epoch 11/20
    20/20 [==============================] - 91s 5s/step - loss: 0.6247 - accuracy: 0.7653 - val_loss: 1.0065 - val_accuracy: 0.6875
    Epoch 12/20
    20/20 [==============================] - 92s 5s/step - loss: 0.5695 - accuracy: 0.7469 - val_loss: 0.7891 - val_accuracy: 0.7000
    Epoch 13/20
    20/20 [==============================] - 93s 5s/step - loss: 0.5589 - accuracy: 0.7671 - val_loss: 0.9282 - val_accuracy: 0.6844
    Epoch 14/20
    20/20 [==============================] - 93s 5s/step - loss: 0.5714 - accuracy: 0.7523 - val_loss: 0.8228 - val_accuracy: 0.7219
    Epoch 15/20
    20/20 [==============================] - 92s 5s/step - loss: 0.4971 - accuracy: 0.7743 - val_loss: 0.7789 - val_accuracy: 0.6812
    Epoch 16/20
    20/20 [==============================] - 75s 4s/step - loss: 0.4710 - accuracy: 0.7831 - val_loss: 0.7657 - val_accuracy: 0.6906
    Epoch 17/20
    20/20 [==============================] - 61s 3s/step - loss: 0.4781 - accuracy: 0.7733 - val_loss: 0.9042 - val_accuracy: 0.6594
    Epoch 18/20
    20/20 [==============================] - 61s 3s/step - loss: 0.4958 - accuracy: 0.8121 - val_loss: 1.0939 - val_accuracy: 0.6406
    Epoch 19/20
    20/20 [==============================] - 61s 3s/step - loss: 0.5523 - accuracy: 0.7698 - val_loss: 0.8393 - val_accuracy: 0.6750
    Epoch 20/20
    20/20 [==============================] - 62s 3s/step - loss: 0.4914 - accuracy: 0.8000 - val_loss: 0.8078 - val_accuracy: 0.6812


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
history = model.fit(train_it, steps_per_epoch=20, validation_data=test_it, validation_steps=10, epochs=20)
```

    Epoch 1/20
    20/20 [==============================] - 68s 3s/step - loss: 0.4814 - accuracy: 0.7812 - val_loss: 0.6448 - val_accuracy: 0.7375
    Epoch 2/20
    20/20 [==============================] - 68s 3s/step - loss: 0.5197 - accuracy: 0.7797 - val_loss: 0.6352 - val_accuracy: 0.7219
    Epoch 3/20
    20/20 [==============================] - 64s 3s/step - loss: 0.4754 - accuracy: 0.7984 - val_loss: 0.7333 - val_accuracy: 0.7000
    Epoch 4/20
    20/20 [==============================] - 68s 3s/step - loss: 0.5086 - accuracy: 0.7893 - val_loss: 0.7059 - val_accuracy: 0.7094
    Epoch 5/20
    20/20 [==============================] - 68s 3s/step - loss: 0.4604 - accuracy: 0.8078 - val_loss: 0.7264 - val_accuracy: 0.6875
    Epoch 6/20
    20/20 [==============================] - 67s 3s/step - loss: 0.5294 - accuracy: 0.7750 - val_loss: 0.6202 - val_accuracy: 0.7344
    Epoch 7/20
    20/20 [==============================] - 68s 3s/step - loss: 0.4691 - accuracy: 0.7984 - val_loss: 0.7077 - val_accuracy: 0.6719
    Epoch 8/20
    20/20 [==============================] - 66s 3s/step - loss: 0.4535 - accuracy: 0.7940 - val_loss: 0.7336 - val_accuracy: 0.6656
    Epoch 9/20
    20/20 [==============================] - 68s 3s/step - loss: 0.4770 - accuracy: 0.8000 - val_loss: 0.7675 - val_accuracy: 0.7219
    Epoch 10/20
    20/20 [==============================] - 67s 3s/step - loss: 0.4839 - accuracy: 0.8047 - val_loss: 0.7697 - val_accuracy: 0.7031
    Epoch 11/20
    20/20 [==============================] - 67s 3s/step - loss: 0.4362 - accuracy: 0.8078 - val_loss: 0.7422 - val_accuracy: 0.7250
    Epoch 12/20
    20/20 [==============================] - 68s 3s/step - loss: 0.4756 - accuracy: 0.8019 - val_loss: 0.8120 - val_accuracy: 0.6969
    Epoch 13/20
    20/20 [==============================] - 72s 4s/step - loss: 0.4289 - accuracy: 0.8176 - val_loss: 0.7332 - val_accuracy: 0.7000
    Epoch 14/20
    20/20 [==============================] - 76s 4s/step - loss: 0.4896 - accuracy: 0.7956 - val_loss: 0.7535 - val_accuracy: 0.6906
    Epoch 15/20
    20/20 [==============================] - 95s 5s/step - loss: 0.4690 - accuracy: 0.7797 - val_loss: 0.8971 - val_accuracy: 0.6625
    Epoch 16/20
    20/20 [==============================] - 96s 5s/step - loss: 0.4789 - accuracy: 0.7875 - val_loss: 0.7222 - val_accuracy: 0.7312
    Epoch 17/20
    20/20 [==============================] - 96s 5s/step - loss: 0.4867 - accuracy: 0.8016 - val_loss: 0.8548 - val_accuracy: 0.6875
    Epoch 18/20
    20/20 [==============================] - 96s 5s/step - loss: 0.4525 - accuracy: 0.8047 - val_loss: 0.7954 - val_accuracy: 0.6781
    Epoch 19/20
    20/20 [==============================] - 95s 5s/step - loss: 0.4089 - accuracy: 0.8192 - val_loss: 0.9348 - val_accuracy: 0.6687
    Epoch 20/20
    20/20 [==============================] - 96s 5s/step - loss: 0.3987 - accuracy: 0.8129 - val_loss: 0.7321 - val_accuracy: 0.6812



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

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

    No handles with labels found to put in legend.



    
![png](output_27_1.png)
    



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




    0




    
![png](output_29_1.png)
    



```python
make_predictions('../sungjin/tooth3/test/20/20683928 김영중 26.png')
```




    0




    
![png](output_30_1.png)
    



```python
make_predictions('../sungjin/tooth3/test/20/20683928 김영중 36.png')
```




    1




    
![png](output_31_1.png)
    



```python
make_predictions('../sungjin/tooth3/test/20/20683928 김영중 46.png')
```




    1




    
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
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}




```python

```
