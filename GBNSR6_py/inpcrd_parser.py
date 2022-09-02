import warnings

from utils import get_numbers
from exceptions import PositionException, PoistionsNumberWarning

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

def parse_traj(lines, num_atoms, mod = 3, replacement ="\n", split_by ='  '):
    # skip lines from the beginning and end of the file if they are not useful

    com_pos = [] # complex position
    new_md = [] # each 3d (mod) position generated in each iteration
    p_counter = 0 # count position when reach 3d (mod)
    frame = 0

    # for each line of the lines
    for l in lines:
        # for each position of the line
        for p in get_numbers(l):
            new_md.append(p)
            p_counter += 1
            if p_counter % mod == 0:
                com_pos.append(new_md)
                p_counter = 0
                new_md = []
                if len(com_pos) == num_atoms:
                    yield com_pos, frame
                    frame += 1
                    com_pos = []

    if p_counter != 0:
        warnings.warn("Number of positions was not as expected, remained (%d)" % p_counter, PoistionsNumberWarning)