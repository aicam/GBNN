import sys, glob, re
#
# usage:
#    python mol_2_sdf.py <output sdf filename>
#
# Consolidates all .mol files in the current directory into a single sdf file.
# Mol files are added to the sdf in a natural sort-like order based on filename.
# Replaces the molecule name field (first line) of each mol file with the the
# mol filename without final file extension, e.g. the sdf entry for the contents
# of guest1.mol has its molecule name field set to guest1. This overwrites anything that
# might already be in the name field of the source mol file.

def natural_sort_key(s):
    ns_pat = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in ns_pat.split(s)] 
 
sdf_filename = sys.argv[1]

mol_filenames = sorted(glob.glob('*.mol'),key=natural_sort_key)

sdf_file = open(sdf_filename,'w')

for filename in mol_filenames:
    print filename
    molecule_name = filename.split('.')[0]
    mol_lines = open(filename).readlines()
    sdf_file.write('%s\n' % molecule_name)
    for line in mol_lines[1:]:
        sdf_file.write(line)
    sdf_file.write('\n$$$$\n')
