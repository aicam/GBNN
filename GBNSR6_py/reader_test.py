import os

mypath = 'tmpMPI'
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
onlyfiles = sorted([onlyfiles[i] for i in range(len(onlyfiles)) if onlyfiles[i].split('.')[-2] == '0' and onlyfiles[i].__contains__('ligand')])
print(onlyfiles)
print(sorted([1, 4, 3]))