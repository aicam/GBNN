import subprocess
import os
from biopandas.pdb import PandasPdb
import pathlib


'''
Get pdb file and run bash script in the get_charges directory
to extract partial charges of each atom
'''
def get_pdb_charges(pdbfile):

    ## Change directory to run bash script
    here = os.path.dirname(os.path.abspath(__file__)) + '/get_charges'
    os.chdir(here)

    print('Here in get_charges ', here)
    print('PDB path ', pdbfile)

    ## TODO: Run script
    # result = subprocess.Popen([os.path.dirname(__file__) + '/get_charges/get_charges.sh', pdbfile], executable="/bin/bash",
    #                           shell=True,
    #                           stdout= subprocess.PIPE,
    #                           stderr= subprocess.PIPE)
    # s, e = result.communicate()
    # while (s.decode("utf-8") == ""):
    #     if (e.decode("utf-8") != ""):
    #         exit(e.decode("utf-8"))
    #     continue
    ## Parse pdb output
    pdb_cat = subprocess.run([here + '/get_charges.sh', pdbfile], capture_output=True).stdout.decode() #s.decode("utf-8").split("\n")
    charges = []
    for l in pdb_cat.split("\n"):
        row = [x for x in l.split(' ') if x != '']
        if len(row) < 4 or row[0] == 'REMARK':
            continue
        charges.append(float(row[-1]))
    return charges
