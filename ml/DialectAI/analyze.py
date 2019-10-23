import numpy as np

data = np.loadtxt('result.txt')
print(data)

data = data.astype(int)
num = 10
stati = np.zeros((num,num))
for (i,j) in data:
    stati[i,j] += 1

stati = stati.astype(int)
def old():
    num = 10
    hardFour = [0,2,5,7]
    new = np.zeros((4,4))
    # res: the error amounts
    res = 0
    for i in range(4):
        for j in range(4):
            ii = hardFour[i]
            jj = hardFour[j]
            print(i,j)
            new[i,j] = stati[ii,jj]
            if i != j:
                res += new[i,j]
    
    new = new.astype(int)
    for i in range(4):
        new[i,i] = 0
    print(new/new.sum() )
    print(res,res/5000)
def new(stati):
    string = ["minnan", "nanchang", 
              "kejia",  "changsha", 
              "shanghai",  "hebei", 
               "hefei",   "shanxi", 
              "sichuan", "ningxia"]

    print(stati)
    # new stati
    for i in range(10):
        stati[i,i] = 0
    stati = np.reshape(stati,(-1))
    arg = np.argsort(stati)
    print(arg)
    arr = np.sort(stati)
    print(arr)
    # inverse
    arg = np.flip(arg,axis=0)
    arr = np.flip(arr,axis=0)
    for i in range(100):
        # from max to min
        # row
        ii = arg[i] // 10
        # column
        jj = arg[i] % 10
        error = sum(arr[:i+1])/10.30
        print(i,',','%.1f%%'%(error),arr[i],string[ii],string[jj])

#new(stati)
old()

