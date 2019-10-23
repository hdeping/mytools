import numpy as np

data = np.loadtxt('result.txt')
print(data)

data = data.astype(int)
stati = np.zeros((4,4))
for (i,j) in data:
    stati[i,j] += 1
print(stati)
