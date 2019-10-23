#coding=utf-8
import openbabel as ob
import pybel as pb
from optparse import OptionParser
import os
import subprocess
import pymysql

# Break the bond in ring and generate the radical fragment
def BreakRing(mol):
	ring_frag = ob.OBMol()
	for atom in ob.OBMolAtomIter(mol):
		ring_frag.AddAtom(atom)
	for bond in ob.OBMolBondIter(mol):
		ring_frag.AddBond(bond)
	return ring_frag

# Determinate the second-end atom
def IsNearTerminal(atom):
	n = 0
	if atom.GetSpinMultiplicity() == 0:
		for _atom in ob.OBAtomAtomIter(atom):
			if _atom.GetType() in ["H", "F", "Cl", "Br", "I"]:
				continue
			else:
				n = n + 1
		return n == 1
	else: 
		return False

# FillAtom() is a function to find all the atom in fragment
def FillAtom(mol, atom, frag, fatom_idx):
	# mol, frag -- Class OBMol
	# atom -- Class OBAtom
	# fatom_idx -- Record the atom has existed
	frag.AddAtom(atom)
	fatom_idx.append(atom.GetIdx())
	if atom.GetValence == 0:
		return frag
	elif IsNearTerminal(atom):
		for _atom in ob.OBAtomAtomIter(atom):
			index = _atom.GetIdx()
			if index in fatom_idx: continue
			frag.AddAtom(_atom)
			fatom_idx.append(index)
		return frag
	else:
		for _atom in ob.OBAtomAtomIter(atom):
			index = _atom.GetIdx()
			if index in fatom_idx: continue
			elif _atom.GetValence == 1:
				frag.AddAtom(_atom)
				fatom_idx.append(index)
			else:
				FillAtom(mol, _atom, frag, fatom_idx)

# BondLink() is a function to bonding the atoms in the fragment 
def FragBondLink(frag, fatom_idx,mol_bond,NumAtomsNoH):
	IdxDict = {}
	for i in range(len(fatom_idx)):
		IdxDict[fatom_idx[i]] = i+1
	
	n = 0
	for j in range(NumAtomsNoH,0,-1):
		if j in fatom_idx:
			n = j
			break
	for BAtom, EAtom, BO in mol_bond:
		if BAtom > n:
			break
		try:
			frag.AddBond(IdxDict.get(BAtom), IdxDict.get(EAtom), BO)
		except:
			continue


# Simplify is a function to remove the same fragment pair
def SimplifyLs(ls1, ls2):
	ls = zip(ls1,ls2)
	for i in range(len(ls)):
		a,b = ls[i]
		if len(a)<=len(b):
			continue
		else:
			ls[i] = (b,a)
	ls.sort()
	i = 0
	while i < len(ls)-1 :
		if ls[i] == ls[i+1]:
			ls.remove(ls[i])
		else:
			i += 1
	return ls


# remove the duplicate fragments
def DelDuplFrag(frag_list):
        Frag_inchi = []
        i = 0
        while i < len(frag_list):
                try:
                        x_inchi = pb.readstring("smi", frag_list[i][0]).write("inchi")
                except:
                        x_inchi = " "
                y_inchi = pb.readstring("smi", frag_list[i][1]).write("inchi")
                if (x_inchi, y_inchi) in Frag_inchi:
                        frag_list.pop(i)
                else:
                        Frag_inchi.append((x_inchi,y_inchi))
                        i = i+1


# obtain gjf file
def Smi2gjf(smi):
	with open('mol.smi', 'w') as file:
		file.write(smi)
	mol = ob.OBMol()
	obConversion = ob.OBConversion()
	obConversion.SetInAndOutFormats('smi', 'gjf')
	obConversion.ReadString(mol, smi)
	mol.AddHydrogens()
	n = mol.NumAtoms()
	charge = mol.GetTotalCharge()
	spin = mol.GetTotalSpinMultiplicity()
	cont = subprocess.check_output('obgen mol.smi', shell = True)
	cont_ls = cont.split('\n')
	cord_gstyle = []
	for line in cont_ls[4:4+n]:
		cordline = line[1:33].split(' ')
		i = 0
		while i < len(cordline):
			if cordline[i] == '':
				cordline.pop(i)
			else:
				i = i + 1
		cord_gstyle.append('{:<2}           {:>7}         {:>7}        {:>7}'.format(cordline[3], cordline[0], cordline[1], cordline[2]))		
	with open('temp1.gjf', 'w') as file:
		file.write('\n' + str(charge)+'  '+str(spin)+'\n')
		for line in cord_gstyle:
			file.write(line + '\n')


# complete the Gaussian input file and you can change the calculation methods
def AdGJF(smi, filename):
	try:
		with open('temp1.gjf', 'r') as _gjf:
			cont = _gjf.readlines()
		os.remove('temp1.gjf')
		# We need to change the file name to make sure the Gaussian process
		# can run appropriately.
		sdd_elem = []
		g2d2p_elem = []
		elements_heavy = 'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn'
		for line in cont[2:]:
			temp = line.split(' ')
#			if temp[0] == 'I':
			if temp[0] in elements_heavy:
				sdd_elem.append(temp[0])
			else:
				g2d2p_elem.append(temp[0])
		with open(filename + '.gjf', 'w') as gjf:
                        if len(sdd_elem) == 0:
                            gjf.writelines(["%nproc=14\n", "%mem=2000mb\n", "%chk="+filename+".chk\n"])
                            gjf.writelines("#n B3LYP 6-31G(d,p) scfcyc=300 empiricaldispersion=gd3 nosymm scf=xqc  opt optcyc=300\n\n")
                            gjf.writelines("we have Title card: ! "+smi+'\n')
                            gjf.writelines(cont)
                            gjf.writelines('\n\n\n\n')
                        else:
                            gjf.writelines(["%nproc=14\n", "%mem=2000mb\n", "%chk="+filename+".chk\n"])
                            gjf.writelines("#n B3LYP gen pseudo=read scfcyc=300 empiricaldispersion=gd3 nosymm scf=xqc  opt optcyc=300\n\n")
                            gjf.writelines("we have Title card: ! "+smi+'\n')
                            gjf.writelines(cont)
                            gjf.writelines('\n')
                            gjf.writelines(" ".join(list(set(sdd_elem))))
                            gjf.writelines('\n'+'SDD\n'+'****\n')
                            if len(g2d2p_elem) == 0:
                                gjf.writelines('\n')
                                gjf.writelines(" ".join(list(set(sdd_elem))))
                                gjf.writelines('\n'+'SDD\n\n')
                            else:
#                                gjf.writelines(" ".join(list(set(g2d2p_elem[:-1]))))
                                gjf.writelines(" ".join(list(set(g2d2p_elem[:]))))
                                gjf.writelines(' 0\n'+'6-31G(d,p)\n'+'****\n\n')
                                gjf.writelines(" ".join(list(set(sdd_elem))))
                                gjf.writelines('\n'+'SDD\n\n')
                            gjf.writelines('\n\n\n\n')

			gjf.writelines('--Link1--\n')
			gjf.writelines(["%nproc=14\n", "%mem=3000mb\n", "%chk="+filename+".chk\n"])
			gjf.writelines("#n B3LYP chkbasis scfcyc=300 empiricaldispersion=gd3 nosymm scf=xqc freq geom=allcheck guess=read\n\n")
#			gjf.writelines("we have Title card: ! "+smi+'\n')
#			gjf.writelines(cont[0:2])
			gjf.writelines("\n\n\n\n\n")

			gjf.writelines('--Link1--\n')
                        if len(sdd_elem) == 0:
                            gjf.writelines(["%nproc=14\n", "%mem=5000mb\n", "%chk="+filename+".chk\n"])
                            gjf.writelines("#n B3LYP 6-311+G(2d,2p) scfcyc=300 nosymm empiricaldispersion=gd3 scf=xqc geom=allcheck guess=read\n\n")
#                           gjf.writelines("we have Title card: ! "+smi+'\n')
#                           gjf.writelines(cont[0:2])
                            gjf.writelines('\n\n\n\n')
                        else:
                            gjf.writelines(["%nproc=14\n", "%mem=5000mb\n", "%chk="+filename+".chk\n"])
                            gjf.writelines("#n B3LYP gen pseudo=read scfcyc=300 empiricaldispersion=gd3 nosymm scf=xqc geom=allcheck guess=read\n\n")
#                           gjf.writelines("we have Title card: ! "+smi+'\n')
#                           gjf.writelines(cont[0:2])
                            gjf.writelines(" ".join(list(set(sdd_elem))))
                            gjf.writelines('\n'+'SDD\n'+'****\n')
                            if len(g2d2p_elem) == 0:
                                gjf.writelines('\n')
                                gjf.writelines(" ".join(list(set(sdd_elem))))
                                gjf.writelines('\n'+'SDD\n\n')
                            else:
#                                gjf.writelines(" ".join(list(set(g2d2p_elem[:-1]))))
                                gjf.writelines(" ".join(list(set(g2d2p_elem[:]))))
                                gjf.writelines(' 0\n'+'6-311+G(2d,2p)\n'+'****\n\n')
                                gjf.writelines(" ".join(list(set(sdd_elem))))
                                gjf.writelines('\n'+'SDD\n\n')
                            gjf.writelines('\n\n\n\n')
                            

	except:
		print "Sorry, no such file in directory! Please check!"


def get_frag_tbl(MolInchi,filename, frag_list):
	with open(filename + '_FragList' +'.txt', 'a') as file:
		for i in range(len(frag_list)):
			file.write(MolInchi+"\t"+frag_list[i][0]+"\t"+frag_list[i][1]+"\n")


def mol_in_db(mol):
	conn1 = pymysql.connect("localhost", "shlchen", "63236985", "bde_b3")
	cur1 = conn1.cursor()
	stat_for_query = "select mol_smi from BDE_B3 where mol_smi = \"{}\"".format(mol)
	flag = cur1.execute(stat_for_query)
	conn1.close()
	return flag != 0


def IsInDB(frag):
#	conn1 = pymysql.connect("localhost", "ljy", "frined122", "bde")
        conn1 = pymysql.connect("localhost", "shlchen", "63236985", "bde_b3")
	cur1 = conn1.cursor()
	if frag == "":
		conn1.close()
		return True
    	else:
        	stat_for_query = "select E_CORR from Frag_B3 where frag_smi = \"{}\"".format(frag)
		flag = cur1.execute(stat_for_query)
		conn1.close()
		return flag != 0


def FragInfo_filler():
#	conn = pymysql.connect("localhost", "ljy", "frined122", "bde")
        conn = pymysql.connect("localhost", "shlchen", "63236985", "bde_b3")
	cur = conn.cursor()
	with open("info.txt", "r") as file:
		for line in file:
			datalist = line.strip().split("\t")
			if len(datalist) == 1:
				continue
			else:
				frag = datalist[0]
				e_corr = datalist[1]
				zpe = datalist[2]
                                corr_e = datalist[3]
				corr_h = datalist[4]
				corr_g = datalist[5]
				sezpe = datalist[6]
				su = datalist[7]
				sh = datalist[8]
				sg = datalist[9]
				try:
					cmd_insert = '''insert ignore into Frag_B3 (frag_smi, E_CORR, ZPE, Corr2E, Corr2H, Corr2G, SeZPE, SU, SH, SG) 
					values (\"{}\", {:.10f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f})'''.format(frag, float(e_corr), float(zpe)
						, float(corr_e), float(corr_h), float(corr_g), float(sezpe), float(su), float(sh), float(sg))
					cur.execute(cmd_insert)
					conn.commit()
				except:
					conn.rollback()
	conn.close()				


def dbFiller(mol_smi, frag1_smi, frag2_smi):
	inchikey = pb.readstring('smi', mol_smi).write('inchikey').replace('\n', '')
#	conn2 = pymysql.connect("localhost", "ljy", "frined122", "bde")
        conn2 = pymysql.connect("localhost", "shlchen", "63236985", "bde_b3")
	cur2 = conn2.cursor()
	stat_for_frag = "select E_CORR, Corr2H from Frag_B3 where frag_smi in (\"{}\", \"{}\")".format(frag1_smi,frag2_smi)
        molcount = cur2.execute(stat_for_frag)
        query_result = cur2.fetchall()
        if molcount == 1: 
		if frag1_smi == '':
			energy1 = 0
			energy2 = query_result[0][0] + query_result[0][1]
		else:
			energy1 = energy2 = query_result[0][0] + query_result[0][1]
        else:
		energy1 = query_result[0][0] + query_result[0][1]
		energy2 = query_result[1][0] + query_result[1][1]
        stat_for_mol = "select E_CORR, Corr2H from Frag_B3 where frag_smi = \"{}\"".format(mol_smi)
        cur2.execute(stat_for_mol)
        result = cur2.fetchall()
        mol_energy = result[0][0] + result[0][1]
        BDE = (energy1 + energy2 - mol_energy) * 627.509474
#	stat_for_insert = "insert ignore into BDE_CCSD (inchikey, mol_smi, frag1_smi, frag2_smi, BDE) values (\"{}\", \"{}\", \"{}\", \"{}\", {:.10f})".format(inchikey, mol_smi, frag1_smi, frag2_smi, BDE)
        stat_for_insert = "insert ignore into BDE_B3 (inchikey, mol_smi, frag1_smi, frag2_smi, BDE) values (\"{}\", \"{}\", \"{}\", \"{}\", {:.10f})".format(inchikey, mol_smi, frag1_smi, frag2_smi, BDE)
	cur2.execute(stat_for_insert)
	conn2.commit()
	conn2.close()


def main():
	# set commandline mode
	# specify the molecule file from the command-line 
	optparser = OptionParser(usage="BDEdb [-i|--input <filename>]")
	optparser.add_option("-i", "--input", action="store", type="string", dest="filename", help='''input molecule file with openbabel accepted formats and the process can generator the data about each bond's BDE''')
	options, args = optparser.parse_args()
	filename = options.filename
	label, inp_flt = os.path.splitext(filename)
	print "Making fragments for the molecule in {}\n".format(filename)
	filetype = inp_flt[1:]
	
	#file's type conversion and generate a 3D builder
	obConversion = ob.OBConversion()
	obConversion.SetInAndOutFormats(filetype, "gjf")

	# create a molecule container
	mol = ob.OBMol()
	obConversion.ReadFile(mol, filename)
	NumAtomsNoH = mol.NumAtoms()
	NumOfBonds = mol.NumBonds()
	MolSmi = pb.Molecule(mol).write("smi").replace('\t\n','')
	if mol_in_db(MolSmi):
		print "Great! It's already in database."
		return 0
	mol.AddHydrogens()

	# get the bond information
	MolBond = []
	ChainBond = []
	for bond in ob.OBMolBondIter(mol):	
		a = bond.GetBeginAtomIdx()
		b = bond.GetEndAtomIdx()
		bo = bond.GetBondOrder()
		MolBond.append((a,b,bo))
		if bond.IsInRing() or bond.IsAromatic():
			continue
		ChainBond.append(bond.GetIdx())
	MolBond.sort()
	NumOfRingBond = len(MolBond) - len(ChainBond)

	# prepare to store fragment's SMILES
	ListOfFrag1 = []
	ListOfFrag2 = []
	for BondIdx in range(len(MolBond)):
		obConversion.ReadFile(mol, filename)
		mol.AddHydrogens()

		Frag1 = ob.OBMol()
		Frag1_idx = []
		Frag2 = ob.OBMol()
		Frag2_idx = []

		# find the begin atom and the end atom of the 
		# breaking bond. 
		a1 = mol.GetBond(BondIdx).GetBeginAtom()
		a2 = mol.GetBond(BondIdx).GetEndAtom()
		breakBO = mol.GetBond(BondIdx).GetBondOrder()

		# homolysis, create radical 
		a1.SetSpinMultiplicity(breakBO+1)
		a2.SetSpinMultiplicity(breakBO+1)
		mol.DeleteBond(mol.GetBond(BondIdx))

		if BondIdx in ChainBond:
			FillAtom(mol, a1, Frag1, Frag1_idx)
			FillAtom(mol, a2, Frag2, Frag2_idx)

			FragBondLink(Frag1, Frag1_idx, MolBond, NumAtomsNoH)
			FragBondLink(Frag2, Frag2_idx, MolBond, NumAtomsNoH)

			frag1_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
			frag2_smi = pb.Molecule(Frag2).write("smi").replace('\t\n','')
		else:
			Frag1 = BreakRing(mol)
			frag2_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
			frag1_smi = ''

		ListOfFrag1.append(frag1_smi)
		ListOfFrag2.append(frag2_smi)
	ListOfFrag = SimplifyLs(ListOfFrag1,ListOfFrag2)
	DelDuplFrag(ListOfFrag)
	print ListOfFrag
	#get_frag_tbl(MolSmi,label, ListOfFrag)
	# Output Gaussian input file
	Frags = [ element for item in ListOfFrag for element in item ]
	print Frags
	Frags_check = []
	Smi2gjf(MolSmi)
	AdGJF(MolSmi, "frag00")	
	for i in range(len(Frags)):
		frag = Frags[i]
		if frag in Frags_check:
			continue
		else:
			if IsInDB(frag) == False:
				Smi2gjf(frag)
				AdGJF(frag, "frag" + str(i))
				Frags_check.append(frag)
	
#	subprocess.call('bash SendCal', shell = True)
        subprocess.call('bash SendCal.sh', shell = True)
        subprocess.call('bash check_normal.sh '+label+'', shell = True)
	subprocess.call('bash OFC_b3.sh', shell = True)
	if not os.path.exists('g09-out'):
            os.makedirs('g09-out')

        subprocess.call('for file in frag*; do mv ./$file ./g09-out/'+label+'_$file; done', shell = True)
	FragInfo_filler()
	for i in range(len(ListOfFrag)):
		frag1 = ListOfFrag[i][0]
		frag2 = ListOfFrag[i][1]
		dbFiller(MolSmi, frag1, frag2)	

        os.remove('info.txt')
        os.remove('mol.smi')

main()

