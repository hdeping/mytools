import numpy as np

data = np.loadtxt('result.txt')
print(data)

data = data.astype(int)
num = 6
stati = np.zeros((num,num))
for (i,j) in data:
    stati[i,j] += 1
print(stati)
