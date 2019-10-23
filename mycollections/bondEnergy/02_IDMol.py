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
    

    stati = 0
    for id in molecules:
        mols = molecules[id]
        length = len(mols)
        # judge if all the molecules with the same ID
        # are equal to each other too.
        if length > 1:
            count = 0
            for i in range(length):
                if mols[0] != mols[i]:
                    print(stati,"wrong here!!!!")
                    for mol in mols:
                        print(id,mol)

                    stati += 1
                    break

                    
    print("error",stati)
            



testId()
#testMol()
