import numpy as np

data = np.loadtxt('result.txt')
#print(data) 
data = data.astype(int)
data = np.reshape(data,(-1,6,2))
num = 10 
stati = np.zeros((num,num))
stati = stati.astype(int)
size = len(data)
# get the majorityVote
target = data[:,0,0]
majority = []

data = data[:,:2,:]
vote_num = len(data[0])
print("vote_num ",vote_num)
for i in range(size):
    # get the vote 
    vote = np.zeros(num)
    vote = vote.astype(int)
    for j in range(vote_num):
        ii = data[i,j,1]
        vote[ii] += 1
    # append the vote result
    #print(vote)
    majority.append(np.argmax(vote))

for i in range(size):
    ii = target[i]
    jj = majority[i]
    stati[ii,jj] += 1

print("size ",size)
arr = []
for i in range(num):
    arr.append(stati[i,i])

arr = np.array(arr)
print("stati")

print(stati)

print("precision")
print(arr/500)
print("acc")
print(sum(arr)/1000)
