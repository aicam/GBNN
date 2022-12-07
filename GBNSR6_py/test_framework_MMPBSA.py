import os
from file_parser import parse_traj, read_gbnsr6_output, store_frame_inpcrd
from utils import get_numbers
import subprocess
import pickle

skip = 1

mdcrd_files = []
for x in sorted(os.listdir()):
    if x.endswith(".crd"):
        mdcrd_files.append(x)

all_records_mmgbsa = []
all_records_mmpbsa = []

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


mmgbsa_dic = {}
mmpbsa_dic = {}
for i in range(1, 2):
    print("End frame is (MMPBSA)" + str(i))
    change_inp_endframe(i)
    subprocess.run(['./run_mmpbsa.sh'])
    f = open('FINAL_RESULTS_MMPBSA.dat').readlines()
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

print("MMPBSA")
print(all_records_mmpbsa)
print("MMGBSA")
print(all_records_mmgbsa)
with open('-'.join(mdcrd_files) + "_" + str(skip) + "_mmgbsa.pkl", 'wb') as handle:
    pickle.dump(all_records_mmgbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('-'.join(mdcrd_files) + "_" + str(skip) + "_mmpbsa.pkl", 'wb') as handle:
    pickle.dump(all_records_mmpbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("finished!!")