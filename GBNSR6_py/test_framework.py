import os
from file_parser import parse_traj, read_gbnsr6_output, store_frame_inpcrd
from utils import get_numbers
import subprocess
import pickle

AMBERHOME = os.environ['AMBERHOME'] #'/home/ali/Amber/amber22'

print(AMBERHOME, " Variable is set")

mdcrd_files = []
num_atoms = 12515
num_solvated = 12515
split_index = 9522
skip = 1
traj_path = "./prods"
traj_type = ".crd"

for x in sorted(os.listdir(traj_path)):
    if x.endswith(traj_type):
        mdcrd_files.append(x)


total_number_frames = 0

all_records_gbnsr6 = []

for mdcrd_file in mdcrd_files:
    mdcrd = open(traj_path + '/' + mdcrd_file, 'r')
    lines = mdcrd.readlines()
    g = parse_traj(lines, num_atoms, num_solvated, skip=skip, return_full=True)

    for fr in g:
        gbnsr6 = {}
        gbnsr6['meta'] = mdcrd_file + "_" + str(fr[1])

        # complex
        subprocess.run(['rm', 'complex.inpcrd'])
        subprocess.run(['rm', 'mdout'])
        store_frame_inpcrd(fr[0])
        print("Frame ", total_number_frames, " stored with len ", len(fr[0]))
        print("Running GBNSR6 on complex")
        subprocess.run(
            [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', './prods/6m0j.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
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
        store_frame_inpcrd(fr[0][:split_index])
        print("Running GBNSR6 on receptor")
        subprocess.run(
            [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'prods/receptor.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
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
        store_frame_inpcrd(fr[0][split_index:])
        print("Running GBNSR6 on ligand")
        subprocess.run(
            [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'prods/sars2.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
        new_res = read_gbnsr6_output('mdout')
        gbnsr6['ligand_Etot'] = new_res['Etot']
        gbnsr6['ligand_EKtot'] = new_res['EKtot']
        gbnsr6['ligand_EPtot'] = new_res['EPtot']
        gbnsr6['ligand_EELEC'] = new_res['EELEC']
        gbnsr6['ligand_EGB'] = new_res['EGB']
        gbnsr6['ligand_ESURF'] = new_res['ESURF']

        all_records_gbnsr6.append(gbnsr6)
        total_number_frames += 1

        print(str(total_number_frames) + " (GBNSR6) Finished")


with open('-'.join(mdcrd_files) + "_" + str(skip) + "_gbnsr6.pkl", 'wb') as handle:
    pickle.dump(all_records_gbnsr6, handle, protocol=pickle.HIGHEST_PROTOCOL)

# MMPBSA runner

def change_inp_endframe(i):
    new_lines=[]
    for l in open('mmpbsa.in'):
        if l.__contains__('endframe'):
            new_lines.append('   endframe=' + str(i) + ',\n')
        elif l.__contains__('startframe'):
            if i == 1:
                new_lines.append('   startframe=1,\n')
            else:
                new_lines.append('   startframe=' + str(i - 1) + ',\n')
        else:
            new_lines.append(l)
    f = open('mmpbsa.in', 'w')
    for l in new_lines:
        f.write(l)
    f.close()

all_records_mmpbsa = []

mmpbsa_dic = {}
for i in range(1, total_number_frames):
    print("End frame is (MMPBSA)" + str(i))
    change_inp_endframe(i)
    subprocess.run(['./run_mmpbsa.sh'])
    f = open('FINAL_RESULTS_MMPBSA.dat').readlines()
    gb = ''.join(f).split('GENERALIZED BORN:')[1].split('POISSON BOLTZMANN:')[0]
    mmpbsa_dic = {}
    mmpbsa_dic['meta'] = str(i)
    for title in ['Complex', 'Receptor', 'Ligand']:
        for param in ['EEL', 'EGB', 'ESURF']:
            mmpbsa_dic[title + '_' + param] = get_numbers(gb.split(title)[1].split('TOTAL')[0].split(param)[1].split('\n')[0])[0]
    all_records_mmpbsa.append(mmpbsa_dic)

with open('-'.join(mdcrd_files) + "_" + str(skip) + "_mmpbsa.pkl", 'wb') as handle:
    pickle.dump(all_records_mmpbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Total number of frames ", total_number_frames)