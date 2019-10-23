import numpy as np

data = np.loadtxt('result.txt')
#print(data)

data = data.astype(int)
def deal(data):
    #(200,2000) -> (1000,200,2)
    data = np.reshape(data,(1000,-1,2))
    print(data.shape)
    num = 2
    stati = np.zeros((num,num))
    stati = stati.astype(int)
    size = len(data)
    # get the majorityVote
    target = data[:,0,0]
    majority = []
    print(len(data[0]))
    for i in range(size):
        # get the vote 
        vote = np.zeros(num)
        vote = vote.astype(int)
        for j in range(len(data[0])):
            ii = data[i,j,1]
            vote[ii] += 1
        # append the vote result
        majority.append(np.argmax(vote))
    for i in range(size):
        ii = target[i]
        jj = majority[i]
        stati[ii,jj] += 1
    arr1 = np.sum(stati,axis=1)
    arr2 = np.sum(stati,axis=0)
    arr = [stati[i,i] for i in range(num)]
    print(stati)
    print("precision , recall")
    print(arr/arr1)
    print(arr/arr2)
    print(sum(arr))
deal(data)
