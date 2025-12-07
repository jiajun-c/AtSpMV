import scipy.io
import numpy as np

path = "/staff/chengjiajun/workspace/database/suitsparse/"
name = "dc1"
A = scipy.io.mmread(path + name + "/" + name + ".mtx").tocsr()
D = np.abs(A.diagonal())
R = np.sum(np.abs(A), axis=1).A1 - D
print(np.all(D > R))

# add20 add32 memplus chipcool0 chipcool1 dc1 dc2 LF10 trans5 trans4