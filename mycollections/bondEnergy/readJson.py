import json

filename = "../new_filter2.json"

fp = open(filename,'r')
data = json.load(fp)

def testId():
    # id and molecule
    molecules = {}
    for residue in data:
        id  = data[residue]['ID']
        molecules[id]  = []
    
    for residue in data:
        id  = data[residue]['ID']
        mol = data[residue]['molecule']
        molecules[id].append(mol)
    
    count = 0
    for key in molecules:
        count += 1
    
    print("id count",count)
def testMol():
    molecules = {}
    for residue in data:
        mol = data[residue]['molecule']
        molecules[mol]  = []
    
    for residue in data:
        id  = data[residue]['ID']
        mol = data[residue]['molecule']
        molecules[mol].append(id)
        #print(mol)

    count = 0
    for key in molecules:
        count += 1
    print("molecules count",count)


testId()
#testMol()
