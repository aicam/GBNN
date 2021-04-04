import subprocess
import os
from biopandas.pdb import PandasPdb



'''
Get pdb file and run bash script in the get_charges directory
to extract partial charges of each atom
'''
def get_pdb_charges(pdbfile):

    ## Change directory to run bash script
    os.chdir("get_charges")

    ## Run script
    result = subprocess.Popen(["./get_charges.sh " + pdbfile], executable="/bin/bash",
                              shell=True,
                              stdout= subprocess.PIPE,
                              stderr= subprocess.PIPE)
    s, e = result.communicate()
    while (s.decode("utf-8") == ""):
        if (e.decode("utf-8") != ""):
            exit(e.decode("utf-8"))
        continue

    ## Parse pdb output
    pandaspdb = PandasPdb()
    ppdb_df = pandaspdb.read_pdb_from_list(pdb_lines=s.decode("utf-8").split("\n"))
    os.chdir('..')
    return ppdb_df.df['ATOM']["b_factor"].to_numpy()

