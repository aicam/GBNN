from exceptions import FilePermission

class LEaP:

    def __init__(self, mdin_name):
        self.mdin_name = mdin_name

    ''' in order to generate topology and coordinate files we need to provide LEaP configurations '''
    def create_mdin(self, pdb_path = '/tmp/complex.pdb', output_path = '/tmp'):
        try:
            f = open(self.mdin_name, 'w')
        except:
            raise FilePermission("Opening %s for LEaP configuration failed" % self.mdin_name)

        f.write('source oldff/leaprc.ff99SB\n')
        f.write('set default PBRadii mbondi2\n')
        f.write('source leaprc.water.tip3p\n')
        f.write('com = loadpdb %s\n' % pdb_path)

        if output_path[-1] == '/':
            output_path = output_path[:-1]

        f.write('saveamberparm com %s/com.prmtop %s/com.inpcrd\n' % (output_path, output_path))
        f.write('solvatebox com TIP3PBOX 12.0\n')
        f.write('saveamberparm com %s/com_solvated.prmtop %s/com_solvated.inpcrd' % (output_path, output_path))
        f.write('quit')
        f.close()