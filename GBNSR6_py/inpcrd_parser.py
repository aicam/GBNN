import warnings

from exceptions import PositionException, PoistionsNumberWarning

def parse_raw_coords(lines, mod = 3, replacement = "\n", skip_first = 2, skip_last = 1):
    # skip lines from the beginning and end of the file if they are not useful
    lines = lines[skip_first: len(lines) - skip_last]

    com_pos = [] # complex position
    new_md = [] # each 3d (mod) position generated in each iteration
    counter = 0 # count position when reach 3d (mod)

    # for each line of the lines
    for l in lines:
        # for each position of the line
        for p in l.split('  '):
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