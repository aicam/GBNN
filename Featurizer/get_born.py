import subprocess
import os
from biopandas.pdb import PandasPdb





def get_born(pdbfile):
    result = subprocess.Popen(["./get_charges.sh " + pdbfile], executable="/bin/bash",
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)