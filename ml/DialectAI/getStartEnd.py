def getStartEnd(arr,p):
    # remain length 
    res_length = arr[2] + arr[0] - arr[1]
    # added length
    add_length = int(res_length*p)
    # new length
    new_length = arr[1] - arr[0] + add_length
    if p<0:
        print("p should be positvie!!!!")
        return -1,-1
        
    if new_length > arr[2]:
        #print("length is too long!!!!!")
        return 0,arr[2]
    # half the length
    half_length = add_length // 2
    #print(add_length,half_length)
    # get start and end
    start = arr[0] - half_length
    end = arr[1] + add_length - half_length
    
    # some special conditions
    if start < 0:
        start = 0
        end = new_length
            
    if end > arr[2]:
        end = arr[2]
        start = arr[2] - new_length
    # return the result
    return start,end
#        
#
#a = [310,472,513]
#for i in range(1,11):
#    start,end = getStartEnd(a,i/10)
#    print(start,end)

