import numpy as np

data = np.loadtxt('result.txt')
print(data)

data = data.astype(int)
data = np.reshape(data,(-1,20,2))
num = 7
stati = np.zeros((num,num))
stati = stati.astype(int)
size = len(data)
# get the majorityVote
target = data[:,0,0]
majority = []
for i in range(size):
    # get the vote 
    vote = np.zeros(20)
    vote = vote.astype(int)
    for j in range(20):
        ii = data[i,j,1]
        vote[ii] += 1
    # append the vote result
    majority.append(np.argmax(vote))
for i in range(size):
    ii = target[i]
    jj = majority[i]
    stati[ii,jj] += 1

arr = []
for i in range(num):
    arr.append(stati[i,i])


print(stati)
print(sum(arr)/5000)
