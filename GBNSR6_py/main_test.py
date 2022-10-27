from cliparser import parser
import utils
import file_parser
import subprocess
import os
import pickle
from gbnsr6_runner import run_gbnsr6

args = vars(parser.parse_args())

AMBERHOME = os.environ['AMBERHOME']

print(AMBERHOME, " Variable is set")

mdcrd_files = []

num_atoms = file_parser.get_natom_topology(args['complex_prmtop'])
split_index = file_parser.get_natom_topology(args['receptor_prmtop'])
num_solvated = 0 if not args['is_solvated'] else file_parser.get_natom_topology(args['solvated_prmtop'])
skip = 1
traj_path = args['traj_path']
traj_type = args['traj_type']

print("Default configuration done! %d atoms in complex, %d atoms in receptor" % (num_atoms, split_index))
print(num_atoms, ' ', split_index)
for x in sorted(os.listdir(traj_path)):
    if x.endswith(traj_type):
        mdcrd_files.append(x)

print("Trajectory files are found: ", mdcrd_files)

total_number_frames = 0
all_records_gbnsr6 = []

for mdcrd_file in mdcrd_files:
    mdcrd = open(traj_path + '/' + mdcrd_file, 'r')
    print("Reading %s trajectory file" % mdcrd_file)
    lines = mdcrd.readlines()
    g = file_parser.parse_traj(lines, num_atoms, num_solvated if num_solvated > 0 else num_atoms, skip=skip, return_full=not args['is_solvated'])

    for fr in g:
        gbnsr6 = {}
        gbnsr6['meta'] = mdcrd_file + "_" + str(fr[1])

        # complex
        print("Frame ", total_number_frames, " stored with len ", len(fr[0]))
        print("Running GBNSR6 on complex")
        gbnsr6.update(run_gbnsr6(AMBERHOME, fr[0], args['complex_prmtop'], 'gbnsr6.in', 'Complex'))

        # receptor
        print("Running GBNSR6 on receptor")
        gbnsr6.update(run_gbnsr6(AMBERHOME, fr[0][:split_index], args['receptor_prmtop'], 'gbnsr6.in', 'Receptor'))

        # ligand
        print("Running GBNSR6 on ligand")
        gbnsr6.update(run_gbnsr6(AMBERHOME, fr[0][split_index:], args['ligand_prmtop'], 'gbnsr6.in', 'Ligand'))

        all_records_gbnsr6.append(gbnsr6)
        total_number_frames += 1

        print(str(total_number_frames) + " (GBNSR6) Finished")


with open('-'.join(mdcrd_files) + "_" + str(skip) + "_gbnsr6.pkl", 'wb') as handle:
    pickle.dump(all_records_gbnsr6, handle, protocol=pickle.HIGHEST_PROTOCOL)


all_records_mmpbsa = []
mmpbsa_dic = {}

for i in range(1, total_number_frames):
    print("Frame is (MMPBSA) started " + str(i))
    file_parser.change_inp_endframe(i)
    subprocess.run(['./run_mmpbsa.sh'])
    f = open('FINAL_RESULTS_MMPBSA.dat').readlines()
    gb = ''.join(f).split('GENERALIZED BORN:')[1].split('POISSON BOLTZMANN:')[0]
    mmpbsa_dic = {}
    mmpbsa_dic['meta'] = str(i)
    for title in ['Complex', 'Receptor', 'Ligand']:
        for param in ['EEL', 'EGB', 'ESURF']:
            mmpbsa_dic[title + '_' + param] = utils.get_numbers(gb.split(title)[1].split('TOTAL')[0].split(param)[1].split('\n')[0])[0]
    all_records_mmpbsa.append(mmpbsa_dic)

with open('-'.join(mdcrd_files) + "_" + str(skip) + "_mmpbsa.pkl", 'wb') as handle:
    pickle.dump(all_records_mmpbsa, handle, protocol=pickle.HIGHEST_PROTOCOL)
