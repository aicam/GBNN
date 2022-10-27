import argparse

parser = argparse.ArgumentParser(description="GBNSR6 python version running over trajectory")

group = parser.add_argument_group('Miscellaneous Options')

group.add_argument('-sp', dest='solvated_prmtop', metavar='<Topology File>',
                  help='''Topology file of a fully solvated system. If provided,
                  the atoms specified by <strip_mask> will be stripped from the
                  trajectory file. The complex topology file (-cp) must be
                  consistent with this stripped trajectory''')
group.add_argument('-rp', dest='receptor_prmtop', metavar='<Topology File>',
                  help='''Topology file of the unbound receptor. If omitted (and
                  -lp is omitted, too), a stability calculation with just the
                  complex will be performed.''')
group.add_argument('-lp', dest='ligand_prmtop', metavar='<Topology File>',
                  help='''Topology file of the unbound ligand. If omitted (and
                  -rp is omitted, too), a stability calculation with just the
                  complex will be performed.''')
group.add_argument('-cp', dest='complex_prmtop', metavar='<Topology File>',
                  default='complex_prmtop', help='''Topology file of the
                  bound complex (or the single system for 'stability'
                  calculations)''')

args = vars(parser.parse_args())
print(args)