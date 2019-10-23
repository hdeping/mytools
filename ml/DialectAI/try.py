
import numpy as np
max_frames = [0,1000]

for i in range(10):
    b = np.random.randint(1000)
    if b > max_frames[0]:
        max_frames[0] = b
    if b < max_frames[1]:
        max_frames[1] = b


print(max_frames)
