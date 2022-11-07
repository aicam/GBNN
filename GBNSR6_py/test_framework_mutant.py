import os
from file_parser import parse_traj, read_gbnsr6_output, store_frame_inpcrd
from utils import get_numbers
import subprocess
import pickle
#
# AMBERHOME = os.environ['AMBERHOME'] #'/home/ali/Amber/amber22'
#
# print(AMBERHOME, " Variable is set")
#
num_solvated = 12509
num_atoms_complex = 12509
num_atoms_receptor = 9522
num_atoms_ligand = 2987
num_frames = 50
#
# skip = 1
complex_traj = "./mutants/_MMPBSA_mutant_complex.mdcrd.0"
# receptor_traj = "./mutants/_MMPBSA_mutant_receptor.mdcrd.0"
# ligand_traj = "./mutants/_MMPBSA_mutant_ligand.mdcrd.0"
# traj_type = ".crd"
#
#
# total_number_frames = 0
#
# all_records_gbnsr6 = []
#
#
# # mdcrd = open(traj_path + '/' + mdcrd_file, 'r')
# # lines = mdcrd.readlines()
# print("Reading trajectory files begin")
# pt_complex = parse_traj(complex_traj, num_atoms_complex, num_atoms_complex, skip=skip)
# pt_receptr = parse_traj(receptor_traj, num_atoms_receptor, num_atoms_receptor, skip=skip)
# pt_ligand = parse_traj(ligand_traj, num_atoms_ligand, num_atoms_ligand, skip=skip)
# print("Reading trajectory files finished")
#
# for i in range(50):
#     fr_complex = next(pt_complex)
#     fr_receptor = next(pt_receptr)
#     fr_ligand = next(pt_ligand)
#     gbnsr6 = {}
#     gbnsr6['meta'] = str(i)
#
#     # complex
#     subprocess.run(['rm', 'complex.inpcrd'])
#     subprocess.run(['rm', 'mdout'])
#     store_frame_inpcrd(fr_complex[0])
#     print("Complex: Frame ", total_number_frames, " stored with len ", len(fr_complex[0]))
#     print("Running GBNSR6 on complex")
#     subprocess.run(
#         [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', './mutants/6m0j-606.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
#     new_res = read_gbnsr6_output('mdout')
#     gbnsr6['complex_Etot'] = new_res['Etot']
#     gbnsr6['complex_EKtot'] = new_res['EKtot']
#     gbnsr6['complex_EPtot'] = new_res['EPtot']
#     gbnsr6['complex_EELEC'] = new_res['EELEC']
#     gbnsr6['complex_EGB'] = new_res['EGB']
#     gbnsr6['complex_ESURF'] = new_res['ESURF']
#
#     # receptor
#     subprocess.run(['rm', 'receptor.inpcrd'])
#     subprocess.run(['rm', 'mdout'])
#     store_frame_inpcrd(fr_receptor[0], fp='./receptor.inpcrd')
#     print("Running GBNSR6 on receptor")
#     subprocess.run(
#         [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'mutants/ACE2.prmtop', '-c', 'receptor.inpcrd', '-i', 'gbnsr6.in'])
#     new_res = read_gbnsr6_output('mdout')
#     gbnsr6['receptor_Etot'] = new_res['Etot']
#     gbnsr6['receptor_EKtot'] = new_res['EKtot']
#     gbnsr6['receptor_EPtot'] = new_res['EPtot']
#     gbnsr6['receptor_EELEC'] = new_res['EELEC']
#     gbnsr6['receptor_EGB'] = new_res['EGB']
#     gbnsr6['receptor_ESURF'] = new_res['ESURF']
#
#     # ligand
#     subprocess.run(['rm', 'ligand.inpcrd'])
#     subprocess.run(['rm', 'mdout'])
#     store_frame_inpcrd(fr_ligand[0], fp='./ligand.inpcrd')
#     print("Running GBNSR6 on ligand")
#     subprocess.run(
#         [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'mutants/SARS-606.prmtop', '-c', 'ligand.inpcrd', '-i', 'gbnsr6.in'])
#     new_res = read_gbnsr6_output('mdout')
#     gbnsr6['ligand_Etot'] = new_res['Etot']
#     gbnsr6['ligand_EKtot'] = new_res['EKtot']
#     gbnsr6['ligand_EPtot'] = new_res['EPtot']
#     gbnsr6['ligand_EELEC'] = new_res['EELEC']
#     gbnsr6['ligand_EGB'] = new_res['EGB']
#     gbnsr6['ligand_ESURF'] = new_res['ESURF']
#
#     all_records_gbnsr6.append(gbnsr6)
#     total_number_frames += 1
#
#     print(str(total_number_frames) + " (GBNSR6) Finished")
#
#
# with open(str(num_frames) + "_" + complex_traj.split('/')[-1] + "_" + str(skip) + "_gbnsr6.pkl", 'wb') as handle:
#     pickle.dump(all_records_gbnsr6, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()
#
# # MMPBSA runner

def change_inp_endframe(i):
    new_lines=[]
    for l in open('mutants/mmpbsa.in'):
        if l.__contains__('endframe'):
            new_lines.append('   endframe=' + str(i) + ',\n')
        elif l.__contains__('startframe'):
            if i == 1:
                new_lines.append('   startframe=1,\n')
            else:
                new_lines.append('   startframe=' + str(i - 1) + ',\n')
        else:
            new_lines.append(l)
    f = open('mutants/mmpbsa.in', 'w')
    for l in new_lines:
        f.write(l)
    f.close()

all_records_mmgbsa = []
all_records_mmpbsa = []
mmgbsa_dic = {}
mmpbsa_dic = {}
for i in range(1, 50):
    print("End frame is (MMPBSA)" + str(i))
    change_inp_endframe(i)
    subprocess.run(['./run_mmpbsa_mutant.sh'])
    f = open('FINAL_RESULTS_MMPBSA.dat').readlines()

    ## GB records
    gb = ''.join(f).split('GENERALIZED BORN:')[1].split('POISSON BOLTZMANN:')[0]
    mmgbsa_dic = {}
    mmgbsa_dic['meta'] = str(i)
    for title in ['Complex', 'Receptor', 'Ligand']:
        for param in ['EEL', 'EGB', 'ESURF']:
            mmgbsa_dic[title + '_' + param] = get_numbers(gb.split(title)[1].split('TOTAL')[0].split(param)[1].split('\n')[0])[0]
    all_records_mmgbsa.append(mmgbsa_dic)

    ## PB records
    pb = ''.join(f).split('POISSON BOLTZMANN:')[1].split('Differences (Complex - Receptor - Ligand):')[1]
    mmpbsa_dic = {}
    mmpbsa_dic['meta'] = str(i)
    for param in ['EEL', 'EPB', 'ENPOLAR', 'EDISPER']:
        mmpbsa_dic[param] = get_numbers(pb.split('DELTA G gas')[0].split(param)[1].split('\n')[0])[0]
    all_records_mmpbsa.append(mmpbsa_dic)

with open(str(num_frames) + "_" + complex_traj.split('/')[-1] + "_" + str(skip) + "_mmgbsa.pkl", 'wb') as handle:
    pickle.dump(all_records_mmgbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
with open(str(num_frames) + "_" + complex_traj.split('/')[-1] + "_" + str(skip) + "_mmpbsa.pkl", 'wb') as handle:
    pickle.dump(all_records_mmpbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

print("Total number of frames ", 50)