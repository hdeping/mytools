import json

filename = "../new_filter2.json"

fp = open(filename,'r')
data = json.load(fp)

def testId():
    # id and molecule
    molecules = {}
    for id in data:
        molecules[id]  = []
    
    for id in data:
        mol = data[id]['molecule']
        molecules[id].append(mol)
    
    count = 0
    for key in molecules:
        count += 1
    
    print("id count",count)
def testMol():
    molecules = {}
    for id in data:
        mol = data[id]['molecule']
        molecules[mol]  = []
    
    for id in data:
        mol = data[id]['molecule']
        molecules[mol].append(id)
        print(mol)

    count = 0
    for key in molecules:
        count += 1
    print("molecules count",count)


testId()
testMol()
