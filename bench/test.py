import scipy.io
import numpy as np

path = "/staff/chengjiajun/workspace/database/suitsparse/"
name = "msdoor"
A = scipy.io.mmread(path + name + "/" + name + ".mtx").tocsr()
D = np.abs(A.diagonal())
R = np.sum(np.abs(A), axis=1).A1 - D
print(np.all(D > R))

# add20 add32 memplus chipcool0 chipcool1 dc1 dc2 LF10 trans5 trans4
# cit-HepPh 8689 1090 8052 7135
# cit-HepTh 4887 1578 13116 10354
# ca-HepPh 4869 2000 12064 11666
# ca-HepTh 5049 2060 7944 8076
# web-BerkStan 7308 14329 15957 13333
# web-Google 12807 11854 9534 17276
# wiki-Vote 5451 1134 15777 11804
# wiki-RfA 5127 1785 9843 7953
# email-EuAll 12438 2526 14796 10352
# email-Enron 9716 2381 9404 8713