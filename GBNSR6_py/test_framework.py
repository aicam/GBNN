import os
import time

from file_parser import parse_traj, read_gbnsr6_output, store_frame_inpcrd
from utils import get_numbers
import subprocess
import pickle

AMBERHOME = os.environ['AMBERHOME'] #'/home/ali/Amber/amber22'

print(AMBERHOME, " Variable is set")

mdcrd_files = []
num_atoms = 3862
num_solvated = 42193
split_index = 2621
skip = 1
traj_path = "./"
traj_type = ".mdcrd"

for x in sorted(os.listdir(traj_path)):
    if x.endswith(traj_type):
        mdcrd_files.append(x)
print("Trajectory files ", mdcrd_files)

total_number_frames = 0

all_records_gbnsr6 = []
gbnsr6_frame_times = []

# for mdcrd_file in mdcrd_files:
#     # mdcrd = open(traj_path + '/' + mdcrd_file, 'r')
#     # lines = mdcrd.readlines()
#     print("Reading trajectory files begin")
#     g = parse_traj(traj_path + '/' + mdcrd_file, num_atoms, num_solvated, skip=skip)
#     print("Reading trajectory files finished")
#
#     for fr in g:
#         gbnsr6 = {}
#         gbnsr6['meta'] = mdcrd_file + "_" + str(fr[1])
#
#         # complex
#         subprocess.run(['rm', 'complex.inpcrd'])
#         subprocess.run(['rm', 'mdout'])
#         store_frame_inpcrd(fr[0])
#         print("Frame ", total_number_frames, " stored with len ", len(fr[0]))
#         print("Running GBNSR6 on complex")
#         time_start = time.time()
#         subprocess.run(
#             [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'ras-raf.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
#         gbnsr6_frame_times.append(time.time() - time_start)
#         new_res = read_gbnsr6_output('mdout')
#         gbnsr6['complex_Etot'] = new_res['Etot']
#         gbnsr6['complex_EKtot'] = new_res['EKtot']
#         gbnsr6['complex_EPtot'] = new_res['EPtot']
#         gbnsr6['complex_EELEC'] = new_res['EELEC']
#         gbnsr6['complex_EGB'] = new_res['EGB']
#         gbnsr6['complex_ESURF'] = new_res['ESURF']
#
#         # receptor
#         subprocess.run(['rm', 'complex.inpcrd'])
#         subprocess.run(['rm', 'mdout'])
#         store_frame_inpcrd(fr[0][:split_index])
#         print("Running GBNSR6 on receptor")
#         subprocess.run(
#             [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'ras.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
#         new_res = read_gbnsr6_output('mdout')
#         gbnsr6['receptor_Etot'] = new_res['Etot']
#         gbnsr6['receptor_EKtot'] = new_res['EKtot']
#         gbnsr6['receptor_EPtot'] = new_res['EPtot']
#         gbnsr6['receptor_EELEC'] = new_res['EELEC']
#         gbnsr6['receptor_EGB'] = new_res['EGB']
#         gbnsr6['receptor_ESURF'] = new_res['ESURF']
#
#         # ligand
#         subprocess.run(['rm', 'complex.inpcrd'])
#         subprocess.run(['rm', 'mdout'])
#         store_frame_inpcrd(fr[0][split_index:])
#         print("Running GBNSR6 on ligand")
#         subprocess.run(
#             [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', 'raf.prmtop', '-c', 'complex.inpcrd', '-i', 'gbnsr6.in'])
#         new_res = read_gbnsr6_output('mdout')
#         gbnsr6['ligand_Etot'] = new_res['Etot']
#         gbnsr6['ligand_EKtot'] = new_res['EKtot']
#         gbnsr6['ligand_EPtot'] = new_res['EPtot']
#         gbnsr6['ligand_EELEC'] = new_res['EELEC']
#         gbnsr6['ligand_EGB'] = new_res['EGB']
#         gbnsr6['ligand_ESURF'] = new_res['ESURF']
#
#         all_records_gbnsr6.append(gbnsr6)
#         total_number_frames += 1
#
#         print(str(total_number_frames) + " (GBNSR6) Finished")


# with open("gbnsr6_frame_times.pkl", 'wb') as handle:
#     pickle.dump(gbnsr6_frame_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()
# with open('-'.join(mdcrd_files) + "_" + str(skip) + "_gbnsr6.pkl", 'wb') as handle:
#     pickle.dump(all_records_gbnsr6, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()

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
def get_numbers(line, limit = -1):
    new_digit = ''
    numbers = []
    is_neg = False
    for i in range(len(line)):
        c = line[i]
        if i < len(line) - 1 and c == '-' and line[i + 1].isdigit():
            is_neg = True
        if c.isdigit() or (c == '.' and new_digit != ''):
            new_digit += c
        if new_digit != '' and c != '.' and (i == len(line) - 1 or not c.isdigit()):
            new_num = float(new_digit)
            new_num = -new_num if is_neg else new_num
            numbers.append(new_num)
            if len(numbers) == limit:
                return numbers
            new_digit = ''
    return numbers
def serializer(mod, f, separator_begin, separator_end):
    data = ''.join(f).split(separator_begin)[1].split(separator_end)[0]
    data_dic = {}
    common_params = []
    if mod == 'gb':
        common_params = ['EEL', 'EGB', 'ESURF']
    if mod == 'pb':
        common_params = ['EEL', 'EPB', 'ENPOLAR', 'EDISPER', 'VDWAALS']
    # all
    for title in ['Complex', 'Receptor', 'Ligand', 'Differences (Complex - Receptor - Ligand)']:
        for param in common_params:
            data_dic[title + '_' + param] = get_numbers(data.split(title)[1].split('TOTAL')[0].split(param)[1].split('\n')[0])[0]
        if title == 'Differences (Complex - Receptor - Ligand)':
            for param in ['DELTA G gas', 'DELTA G solv', 'DELTA TOTAL'] + common_params:
                data_dic[title + '_' + param] = get_numbers(data.split(title)[1].split(param)[1].split('\n')[0])[0]
    return data_dic

all_records_mmgbsa = []
all_records_mmpbsa = []
mmpbsa_frame_times = []
mmgbsa_dic = {}
mmpbsa_dic = {}
for i in range(1, 200):#total_number_frames):
    print("End frame is (MMPBSA)" + str(i))
    change_inp_endframe(i)
    time_start = time.time()
    subprocess.run(['./run_mmpbsa.sh'])
    mmpbsa_frame_times.append(time.time() - time_start)
    f = open('FINAL_RESULTS_MMPBSA.dat').readlines()

    ## GB records
    # all_records_mmgbsa.append(serializer('gb', f, 'GENERALIZED BORN:', '-------------------------------------------------------------------------------\n-------------------------------------------------------------------------------'))

    ## PB records
    # all_records_mmpbsa.append(serializer('pb', f, 'POISSON BOLTZMANN:', '-------------------------------------------------------------------------------\n-------------------------------------------------------------------------------'))

with open("mmpbsa_frame_times.pkl", 'wb') as handle:
    pickle.dump(mmpbsa_frame_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

# with open('-'.join(mdcrd_files) + "_" + str(skip) + "_mmgbsa.pkl", 'wb') as handle:
#     pickle.dump(all_records_mmgbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()
# with open('-'.join(mdcrd_files) + "_" + str(skip) + "_mmpbsa.pkl", 'wb') as handle:
#     pickle.dump(all_records_mmpbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()

print("Total number of frames ", total_number_frames)