import subprocess
import os
from biopandas.pdb import PandasPdb





def get_born(pdbfile):
    os.chdir("get_born")
    result = subprocess.run(["./get_born.sh " + pdbfile], executable="/bin/bash",
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    while (result.stdout.decode("utf-8") == ""):
        continue
    pandaspdb = PandasPdb()
    print(result.stdout.decode("utf-8"))
    # TODO: add gbnsr6 parser
    # ppdb_df = pandaspdb.read_pdb_from_list(pdb_lines=result.stdout.decode("utf-8").split("\n"))
    # return ppdb_df.df['ATOM']

print(get_born("~/calstate/amber/ras.pdb"))