import os
import numpy as np
import tensorflow as tf
import keras_applications
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path_dir1 = 'C:\\Users\\LG\\Desktop\\nomask\\face\\'
path_dir2 = 'C:\\Users\\LG\\Desktop\\mask\\face\\'
path_dir3 = 'C:\\Users\\LG\\Desktop\\tukmask\\face\\'
path_dir4 = 'C:\\Users\\LG\\Desktop\\tukmask2\\face\\'

file_list1 = os.listdir(path_dir1)  # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
# file_list3 = os.listdir(path_dir3)
# file_list4 = os.listdir(path_dir4)

file_list1_num = len(file_list1)
file_list2_num = len(file_list2)
# file_list3_num = len(file_list3)
# file_list4_num = len(file_list4)

file_num = file_list1_num + file_list2_num #+ file_list3_num + file_list4_num

# preprocessing image
num = 0;
check_img = np.float32(np.zeros((file_num, 224, 224, 3)))
check_label = np.float64(np.zeros((file_num, 1)))

for img_name in file_list1:
    img_path = path_dir1 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    check_img[num, :, :, :] = x

    check_label[num] = 0  # nomask
    num = num + 1

for img_name in file_list2:
    img_path = path_dir2 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    check_img[num, :, :, :] = x

    check_label[num] = 1  # mask
    num = num + 1

# for img_name in file_list3:
#     img_path = path_dir3 + img_name
#     img = load_img(img_path, target_size=(224, 224))
#
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     check_img[num, :, :, :] = x
#
#     check_label[num] = 0  # tuk-mask
#     num = num + 1


n_elem = check_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)

check_label = check_label[indices]
check_img = check_img[indices]

# train : test = 8 : 2
num_train = int(np.round(check_label.shape[0] * 0.8))
num_test = int(np.round(check_label.shape[0] * 0.2))

train_img = check_img[0:num_train, :, :, :]
test_img = check_img[num_train:, :, :, :]

train_label = check_label[0:num_train]
test_label = check_label[num_train:]


# transfer learning ( ResNet50 )
IMG_SHAPE = (224, 224, 3)

base_model = ResNet152(input_shape=IMG_SHAPE,
                      weights='imagenet',
                      include_top=False)

# Freeze Network
base_model.trainable = False

inputs = tf.keras.Input(IMG_SHAPE)
# Separately from setting trainable on the model, we set training to False
x = base_model(inputs, training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(1,activation = tf.nn.sigmoid)(x)
model = tf.keras.Model(inputs,outputs)
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss ='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_img, train_label, epochs=3, batch_size=16, validation_data=(test_img, test_label))

# Unfreeze Network
base_model.trainable = True

model.compile(optimizer = tf.keras.optimizers.Adam(lr = .00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_img, train_label, epochs=3, batch_size=16, validation_data=(test_img, test_label))

# save model
model.save("model_final.h5")