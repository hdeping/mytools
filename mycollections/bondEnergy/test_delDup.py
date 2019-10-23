
from getsmiles import Simplify

a = ['aa','aa','bb','cc','bb','dd','ee']
b = ['1','1','2','3','2','4','5']
print(len(a),len(b))


res = Simplify(a,b)
print(res)


import numpy as np

res = np.array(res)
print(res[:,0])

