import subprocess
import os
from biopandas.pdb import PandasPdb




def get_pdb_charges(pdbfile):
    os.chdir("get_charges")
    result = subprocess.Popen(["./get_charges.sh " + pdbfile], executable="/bin/bash",
                              shell=True,
                              stdout= subprocess.PIPE,
                              stderr= subprocess.PIPE)
    s, e = result.communicate()
    pandaspdb = PandasPdb()
    # print(s.decode("utf-8"))
    ppdb_df = pandaspdb.read_pdb_from_list(pdb_lines=s.decode("utf-8").split("\n"))
    return ppdb_df.df['ATOM']

print(get_pdb_charges("~/calstate/amber/ras.pdb")["b_factor"])