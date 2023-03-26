import sys, glob, re


def natural_sort_key(s):
    ns_pat = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in ns_pat.split(s)] 
 
mol_filenames = sorted(glob.glob('*.sdf'),key=natural_sort_key)

for filename in mol_filenames:
    print filename
    mol_lines = open(filename).readlines()
    new_file = open(filename,'w')
    line_number = 0
    for line in mol_lines[0:]:
        line_number += 1
        if line_number == 4:
            line = line[0:6] + "  0  0  0  0  0  0  0  0999 V2000\n"
        new_file.write(line)
