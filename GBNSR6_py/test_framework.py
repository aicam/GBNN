import os
from file_parser import parse_traj, read_gbnsr6_output, store_frame_inpcrd
from utils import get_numbers
import subprocess
import pickle

AMBERHOME = '/home/ali/Amber/amber22'

mdcrd_files = []
for x in sorted(os.listdir()):
    if x.endswith(".mdcrd"):
        mdcrd_files.append(x)

num_atoms = 3862
num_solvated = 42193
skip = 1

all_records_gbnsr6 = []

for mdcrd_file in mdcrd_files:
    mdcrd = open(mdcrd_file, 'r')
    lines = mdcrd.readlines()
    g = parse_traj(lines, num_atoms, num_solvated, skip=skip)

    f_count = 0
    for fr in g:
        gbnsr6 = {}
        gbnsr6['meta'] = mdcrd_file + "_" + str(fr[1])

        # complex
        subprocess.run(['rm', 'complex.inpcrd'])
        subprocess.run(['rm', 'mdout'])
        store_frame_inpcrd(fr[0])
        subprocess.run(
            [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'com.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
        new_res = read_gbnsr6_output('mdout')
        gbnsr6['complex_Etot'] = new_res['Etot']
        gbnsr6['complex_EKtot'] = new_res['EKtot']
        gbnsr6['complex_EPtot'] = new_res['EPtot']
        gbnsr6['complex_EELEC'] = new_res['EELEC']
        gbnsr6['complex_EGB'] = new_res['EGB']
        gbnsr6['complex_ESURF'] = new_res['ESURF']

        # receptor
        subprocess.run(['rm', 'complex.inpcrd'])
        subprocess.run(['rm', 'mdout'])
        store_frame_inpcrd(fr[0][:2621])
        subprocess.run(
            [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'ras.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
        new_res = read_gbnsr6_output('mdout')
        gbnsr6['receptor_Etot'] = new_res['Etot']
        gbnsr6['receptor_EKtot'] = new_res['EKtot']
        gbnsr6['receptor_EPtot'] = new_res['EPtot']
        gbnsr6['receptor_EELEC'] = new_res['EELEC']
        gbnsr6['receptor_EGB'] = new_res['EGB']
        gbnsr6['receptor_ESURF'] = new_res['ESURF']

        # ligand
        subprocess.run(['rm', 'complex.inpcrd'])
        subprocess.run(['rm', 'mdout'])
        store_frame_inpcrd(fr[0][2621:])
        subprocess.run(
            [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'raf.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
        new_res = read_gbnsr6_output('mdout')
        gbnsr6['ligand_Etot'] = new_res['Etot']
        gbnsr6['ligand_EKtot'] = new_res['EKtot']
        gbnsr6['ligand_EPtot'] = new_res['EPtot']
        gbnsr6['ligand_EELEC'] = new_res['EELEC']
        gbnsr6['ligand_EGB'] = new_res['EGB']
        gbnsr6['ligand_ESURF'] = new_res['ESURF']

        all_records_gbnsr6.append(gbnsr6)

        f_count += 1
        print(str(f_count) + " Finished")
        ## TODO: remove this line
        if f_count == 2:
            break

with open('-'.join(mdcrd_files) + "_" + str(skip) + "_gbnsr6.pkl", 'wb') as handle:
    pickle.dump(all_records_gbnsr6, handle, protocol=pickle.HIGHEST_PROTOCOL)

# MMPBSA runner

def change_inp_endframe(i):
    new_lines=[]
    for l in open('mmpbsa.in'):
        if l.__contains__('endframe'):
            new_lines.append('   endframe=' + str(i) + ',\n')
        else:
            new_lines.append(l)
    f = open('mmpbsa.in', 'w')
    for l in new_lines:
        f.write(l)
    f.close()

all_records_mmpbsa = []

mmpbsa_dic = {}
for i in range(1, 2):
    print("End frame is " + str(i))
    change_inp_endframe(i)
    subprocess.run(['./run_mmpbsa.sh'])
    f = open('FINAL_RESULTS_MMPBSA.dat').readlines()
    gb = ''.join(f).split('GENERALIZED BORN:')[1].split('POISSON BOLTZMANN:')[0]
    mmpbsa_dic = {}
    for title in ['Complex', 'Receptor', 'Ligand']:
        for param in ['EEL', 'EGB', 'ESURF']:
            mmpbsa_dic[title + '_' + param] = get_numbers(gb.split(title)[1].split('TOTAL')[0].split(param)[1].split('\n')[0])[0]
    all_records_mmpbsa.append(mmpbsa_dic)

with open('-'.join(mdcrd_files) + "_" + str(skip) + "_mmpbsa.pkl", 'wb') as handle:
    pickle.dump(all_records_mmpbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)