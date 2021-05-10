import sys
from sklearn.metrics import f1_score
import numpy as np

result_path = sys.argv[1] # result.npy
GT = sys.argv[2] # "data/y_test.npy

result = np.load(result_path)
y_prediction = np.argmax(result, 1)
y_test = np.load(GT)
y_true = np.squeeze(y_test ,axis=1)

print(f1_score(y_true, y_prediction, average="weighted"))