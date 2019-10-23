import numpy as np

data = np.loadtxt('result.txt')
print(data)

data = data.astype(int)
count = 200
data = np.reshape(data,(-1,200,2))
np.savetxt('predict_target.txt',data[:,:,1],fmt='%d')
num = 2
stati = np.zeros((num,num))
stati = stati.astype(int)
size = len(data)
# get the majorityVote
target = data[:,0,0]
majority = []
for i in range(size):
    # get the vote 
    vote = np.zeros(num)
    vote = vote.astype(int)
    for j in range(count):
        ii = data[i,j,1]
        vote[ii] += 1
    # append the vote result
    majority.append(np.argmax(vote))
for i in range(size):
    ii = target[i]
    jj = majority[i]
    stati[ii,jj] += 1
print(stati)

correct = []
for i in range(num):
    correct.append(stati[i,i])
print(correct)
print(sum(correct))



