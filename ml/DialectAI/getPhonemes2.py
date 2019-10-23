import numpy  as np

# get the binary sequence with a fixed length 8 of a number
def seq2arr(num):
    res = []
    tmp = num
    for i in range(8):
        res.append(tmp%2)
        tmp = tmp // 2
    res = np.array(res)
    #print(res)
    return res

def getPhonemes():

    # ge mlf
    name = "total.mlf"
    data = np.loadtxt(name,dtype=str)
    # get phonemes dictionary
    phonemes_dict = {}
    for i,phoneme in enumerate(data):
        phonemes_dict[phoneme] = seq2arr(i)
    return phonemes_dict
def dealMlf(filename):
    phonemes_dict = getPhonemes()
    data = np.loadtxt(filename,dtype=str)
    #print(data)

    # get the output which is a dictionary
    output_dict = {}
    
    for line in data:
        # add a new key
        if line.startswith('"'):
            # get the sample file name
            line = line.split('/')
            # deliminate the " symbol
            line = line[1]
            # replace 'lab' with 'fb'
            line = line[:-1].replace('lab','fb')
            # line is the current key
            key = line
            output_dict[key] = []
            #print(line)
            continue
        # skip <s> </s>  and . lines  
        if line == '.':
            continue
        if line == '<s>':
            continue
        if line == '</s>':
            continue
        # append the phonemes to the current key
        #print(line,key)
        output_dict[key].append(phonemes_dict[line])
    # deal with the output dictionary
    for line in output_dict:
        output_dict[line] = np.array(output_dict[line])
        #print(output_dict[line].shape)

    return output_dict

#print(phonemes_dict)
# cat `find . -name "*train*mlf"` > train.mlf
# sed -i '/MLF/'d
#filename = "train.mlf"
#output_dict = dealMlf(filename)
#print(output_dict)


