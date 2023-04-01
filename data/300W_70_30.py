import numpy as np
import os

filelist_path = './datasets/300W_LP/files.txt'
filelist = np.loadtxt(filelist_path, dtype=str, delimiter='\n')

n_train = int(len(filelist) * 0.7)

randFlag = np.zeros(len(filelist))
randFlag[:n_train] = 1
randFlag = np.random.permutation(randFlag)

trainlist_path = os.path.join(os.path.dirname(filelist_path), 'train.txt')
testlist_path = trainlist_path.replace('train', 'test')

train_list = []
test_list = []
for i in range(len(filelist)):
    file_path = filelist[i]
    if randFlag[i] == 1:
        train_list.append(file_path)
    elif randFlag[i] == 0:
        test_list.append(file_path)

np.savetxt(trainlist_path, train_list, fmt='%s', delimiter='/n')
np.savetxt(testlist_path, test_list, fmt='%s', delimiter='/n')
