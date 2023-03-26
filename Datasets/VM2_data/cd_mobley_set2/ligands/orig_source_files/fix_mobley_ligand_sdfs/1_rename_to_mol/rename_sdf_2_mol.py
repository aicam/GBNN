import sys, glob, re
import os

def natural_sort_key(s):
    ns_pat = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in ns_pat.split(s)]

sdf_filenames = sorted(glob.glob('*.sdf'),key=natural_sort_key)

for filename in sdf_filenames:
    print filename
    filename_new = filename[:-4] + '.mol'
    os.system('mv ./%s %s' % (filename, filename_new))
