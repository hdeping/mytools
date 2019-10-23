import numpy as np
train_list_file = "../labels/label_train-0-1-2-5.txt"
train_list = np.loadtxt(train_list_file, delimiter=' ', dtype=str)
np.random.shuffle(train_list)
#print(train_list)
np.savetxt(train_list_file+'1',train_list,delimiter=' ',fmt="%s")
print(len(train_list))
