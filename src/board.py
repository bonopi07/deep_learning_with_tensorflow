from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

x = np.arange(20).reshape((10, 2))
x2 = np.arange(10).reshape((5, 2))
y = np.arange(10)
y2 = np.arange(5)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
for (train1_idx, eval1_idx), (train2_idx, eval2_idx) in zip(cv.split(x), cv.split(x2)):
    print('train idx:', train1_idx, train2_idx)
    print('eval idx:', eval1_idx, eval2_idx)
    print('.' * 50)

    print(x[train1_idx])