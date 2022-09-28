import os
from file_parser import parse_traj, read_gbnsr6_output, store_frame_inpcrd
from utils import get_numbers
import subprocess
import pickle


AMBERHOME = os.environ['AMBERHOME'] #'/home/ali/Amber/amber22'
skip = 1
print(AMBERHOME, " Variable is set")

mdcrd_files = []
for x in sorted(os.listdir()):
    if x.endswith(".mdcrd"):
        mdcrd_files.append(x)

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


mmpbsa_dic = {}
for i in range(1, 2):
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