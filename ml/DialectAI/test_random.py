import numpy as np
import random 

random.seed(2000)

a = np.arange(10)
print(a)

for i in range(10):
    random.shuffle(a)
    print(a)
