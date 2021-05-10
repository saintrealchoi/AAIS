import matplotlib.pyplot as plt
import numpy as np

# 각 숫자 라벨별 class

# 0: airplane
# 1: automobile
# 2: bird
# 3: cat
# 4: deer
# 5: dog
# 6: frog
# 7: horse
# 8: ship
# 9: truck


x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

plt.imshow(x_train[0], interpolation="bicubic")
plt.show()
print(y_train[0])
