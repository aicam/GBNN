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
                  complex will be performed.''', default='./prods/receptor.prmtop')
group.add_argument('-lp', dest='ligand_prmtop', metavar='<Topology File>',
                  help='''Topology file of the unbound ligand. If omitted (and
                  -rp is omitted, too), a stability calculation with just the
                  complex will be performed.''', default='./prods/ligand.prmtop')
group.add_argument('-cp', dest='complex_prmtop', metavar='<Topology File>',
                  default='./prods/complex.prmtop', help='''Topology file of the
                  bound complex (or the single system for 'stability'
                  calculations)''')

## Extra
group.add_argument('-is', dest='is_solvated', metavar='Bool',
                  default=True, help='''Is the simulation run over a solvated structure?''')
# group.add_argument('--only-gbnsr6', dest='only_gbnsr6', metavar='Bool',
#                    default=False, help='''Running GBNSR6 only over the trajectory''')
# group.add_argument('--only-mmpbsa', dest='only_mmpbsa', metavar='Bool',
#                    default=False, help='''Running MMPBSA only over the trajectory''')
group.add_argument('--traj-path', dest='traj_path', metavar='Path',
                   default='./prods', help='''Trajectory files path''')
group.add_argument('--traj-type', dest='traj_type', metavar='file type',
                   default='.crd', help='''Trajectory files type''')

# args = vars(parser.parse_args())
# print(args)