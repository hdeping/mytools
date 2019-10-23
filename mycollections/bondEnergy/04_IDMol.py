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
    residues = []
    for id in molecules:
        mols = molecules[id]
        length = len(mols)
        # judge if all the molecules with the same ID
        # are equal to each other too.
        if length > 1:
            count = 0
            for i in range(length):
                if mols[0] != mols[i]:
                    #print(stati,"wrong here!!!!")
                    # get energies
                    for residue in data:
                        key_id  = data[residue]['ID']
                        if key_id == id:
                            erg     = data[residue]['energy']
                            #print(key_id,erg)
                            residues.append(residue)

                    #for mol in mols:
                    #    print(id,mol)

                    stati += 1
                    break

                    
    print("error",stati)
    # delete the residues
    for residue in residues:
        del data[residue]
    database = json.dumps(data,indent=4)
    
    #### write the data
    print("write the data to json")
    filename = "new_filter3.json"
    fp = open(filename,'w')
    fp.write(database)
    fp.close()
            

    


testId()
#testMol()
