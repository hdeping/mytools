import numpy as np

data = np.loadtxt('result.txt')
#print(data)

data = data.astype(int)
def getStati():
    num = 10
    stati = np.zeros((num,num))
    stati = stati.astype(int)
    for (i,j) in data:
        stati[i,j] += 1
    # 3 4 6 8 9
    arr = np.array([3,4,6,8,9])
    size = len(arr)
    index = 0
    new_stati = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            ii = arr[i]
            jj = arr[j]
            new_stati[i,j] = stati[ii,jj]
            index += 1
    
    print(index)
    stati = new_stati
    print(stati)
    return stati


stati = getStati()


num = len(stati)
arr1 = np.sum(stati,axis=1)
arr2 = np.sum(stati,axis=0)

predict = np.zeros((num))
for i in range(num):
    predict[i] = stati[i,i]
    
print(arr1)
print(arr2)
print("Precision")
precision = predict / arr1
print(precision)
print("Recall")
recall = predict / arr2
print(recall)
f1 = precision * recall
print("f1")
print(f1)

