import warnings

from utils import get_numbers, generate_inpcrd_num, add_zeros
from exceptions import PositionException, PoistionsNumberWarning, FilePermission
import numpy as np

def parse_raw_inpcrd(lines, mod = 3, replacement ="\n", skip_first = 2, skip_last = 1, split_by ='  '):
    # skip lines from the beginning and end of the file if they are not useful
    lines = lines[skip_first: len(lines) - skip_last]

    com_pos = [] # complex position
    new_md = [] # each 3d (mod) position generated in each iteration
    counter = 0 # count position when reach 3d (mod)

    # for each line of the lines
    for l in lines:
        # for each position of the line
        for p in l.split(split_by):
            p_trim = p.replace(replacement, '')
            # if we can not find any position skip the rest of the loop
            if p_trim == '':
                continue
            try:
                p_1d = float(p_trim)
            except:
                raise PositionException("Invalid value for a postion (%s)!" % p_trim)

            new_md.append(p_1d)
            counter += 1
            if counter % mod == 0:
                com_pos.append(new_md)
                counter = 0
                new_md = []
    if counter != 0:
        warnings.warn("Number of positions was not as expected, remained (%d)" % counter, PoistionsNumberWarning)
    return com_pos

def parse_traj(f, num_atoms, num_total, mod = 3, skip = 1, end_frame = 9999, return_full = False):

    com_pos = [] # complex position
    new_md = [] # each 3d (mod) position generated in each iteration
    p_counter = 0 # count position when reach 3d (mod)
    skip_counter = 0 # skip frames based on skip in input
    frame = 0

    with open(f, 'r') as infile:
        # for each line of the lines
        for l in infile:
            # for each position of the line
            for p in get_numbers(l):
                new_md.append(p)
                p_counter += 1
                if p_counter % mod == 0:
                    com_pos.append(new_md)
                    p_counter = 0
                    new_md = []
                    if len(com_pos) == num_total:
                        frame += 1
                        if frame > end_frame:
                            return
                        if frame % skip != 0:
                            com_pos = []
                            continue
                        ## we store all atoms in solvated form but return only the dry positions
                        if return_full:
                            yield com_pos, frame
                        else:
                            yield com_pos[:num_atoms], frame
                        com_pos = []

    # if p_counter != 0:
    #     warnings.warn("Number of positions was not as expected, remained (%d)" % p_counter, PoistionsNumberWarning)

def store_frame_inpcrd(coords, per_line = 6, fp = './complex.inpcrd'):
    try:
        f = open(fp, 'w')
    except:
        raise FilePermission("To store inpcrd frames, the program needs to have access to %s" % fp)

    f.write('default_name\n')
    white_spaces = ''.join([' ' for i in range(6 - len(coords))])
    f.write(white_spaces + str(len(coords)) + '\n')

    counter_p = 0 # counts number of positions already written in a line

    for coordMD in coords:
        for coord1D in coordMD:
            f.write(generate_inpcrd_num(coord1D))
            counter_p += 1
            if counter_p % per_line == 0:
                f.write("\n")

    f.close()

def store_single_frame_mdcrd(coords, per_line = 10, fp = './single.mdcrd'):
    try:
        f = open(fp, 'w')
    except:
        raise FilePermission("To store inpcrd frames, the program needs to have access to %s" % fp)

    f.write('\n')

    counter_p = 0 # counts number of positions already written in a line

    for coordMD in coords:
        for coord1D in coordMD:
            if coord1D < 0:
                f.write(' ' + add_zeros(np.around(coord1D, 3)))
            else:
                f.write('  ' + add_zeros(np.around(coord1D, 3)))
            counter_p += 1
            if counter_p % per_line == 0:
                f.write("\n")

    f.close()
def read_gbnsr6_output(path):
    f = open(path)
    lines = f.readlines()
    start = 0
    res_lines = []
    for i in range(len(lines)):
        if lines[i].__contains__('FINAL RESULT'):
            start = i + 3
            break
    for l in lines[start:]:
        if not l.__contains__('-------'):
            res_lines.append(l.replace('\n', ''))
        else:
            break
    i = 1
    result = {}
    l = res_lines[i]
    result['Etot'] = get_numbers(l.split('EKtot')[0])[0]
    result['EKtot'] = get_numbers(l.split('EKtot')[1], 1)[0]
    result['EPtot'] = get_numbers(l.split('EPtot')[1])[0]
    i += 3
    l = res_lines[i]
    result['EELEC'] = get_numbers(l.split('EGB')[0])[0]
    result['EGB'] = get_numbers(l.split('EGB')[1], 1)[0]
    i += 1
    l = res_lines[i]
    result['ESURF'] = get_numbers(l.split('=')[1])[0]

    return result

## TODO: check consistency of `FLAG POINTERS`
def get_natom_topology(f):
    lines = open(f, 'r').readlines()
    natom = 0
    for i in range(len(lines)):
        if lines[i].__contains__('FLAG POINTERS'):
            natom = get_numbers(lines[i + 2])[0]
    return int(natom)

def change_inp_endframe(i, fm = 'mmpbsa.in'):
    new_lines=[]
    for l in open(fm):
        if l.__contains__('endframe'):
            new_lines.append('   endframe=' + str(i) + ',\n')
        elif l.__contains__('startframe'):
            if i == 1:
                new_lines.append('   startframe=1,\n')
            else:
                new_lines.append('   startframe=' + str(i - 1) + ',\n')
        else:
            new_lines.append(l)
    f = open(fm, 'w')
    for l in new_lines:
        f.write(l)
    f.close()
