import subprocess
import os





def get_pdb_born(pdbfile):
    os.chdir("get_born")
    result = subprocess.run(["./get_born.sh " + pdbfile], executable="/bin/bash",
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    while (result.stdout.decode("utf-8") == ""):
        if (result.stderr.decode("utf-8") != ""):
            exit(result.stderr.decode("utf-8"))
        continue
    # print(result.stdout.decode("utf-8"))
    # TODO: Ask prof, rinv is consistent?
    lines = result.stdout.decode("utf-8").split("\n")
    B = []
    for l in lines:
        if l[:4] != "rinv":
            continue
        B.append(float(list(filter(None, l.split(' ')))[2].replace('\n', '')))
    os.chdir('..')
    # ppdb_df = pandaspdb.read_pdb_from_list(pdb_lines=result.stdout.decode("utf-8").split("\n"))
    # return ppdb_df.df['ATOM']
    return B
# print(get_born("~/calstate/amber/ras.pdb"))