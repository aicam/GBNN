import subprocess
import os




'''
Get pdb file and run bash script in the get_born directory
to extract effective born radii of each atom
'''
def get_pdb_born(pdbfile):

    ## Change directory to run bash script
    os.chdir("get_born")

    ## Run script
    result = subprocess.run(["./get_born.sh " + pdbfile], executable="/bin/bash",
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    while (result.stdout.decode("utf-8") == ""):
        if (result.stderr.decode("utf-8") != ""):
            exit(result.stderr.decode("utf-8"))
        continue

    # TODO: Ask prof, rinv is consistent?
    lines = result.stdout.decode("utf-8").split("\n")
    B = []

    # Extract inverted bor radii
    for l in lines:
        if l[:4] != "rinv":
            continue
        B.append(float(list(filter(None, l.split(' ')))[2].replace('\n', '')))
    os.chdir('..')

    return B