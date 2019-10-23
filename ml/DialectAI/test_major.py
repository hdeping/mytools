
import numpy as np

data = np.loadtxt('result.txt')
data = data[:,:2]
count = 0

for line  in data:
    if line[0] == line[1]:
        count = count + 1

print(count)
