

def parse_raw_coords(lines, mod = 3, replacement = "\n", skip_first = 2, skip_last = 1):
    # skip lines from the beginning and end of the file if they are not useful
    lines = lines[skip_first: len(lines) - skip_last]

    # for each line of the lines
    for l in lines:
        # for each position of the line
        for p in l.split('  '):
            p_trim = p.replace(replacement, '')
            # if we can not find any position skip the rest of the loop
            if p_trim == '':
                continue

