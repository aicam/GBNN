from exceptions import FilePermission
from subprocess import Popen
class LEaP:

    def __init__(self, mdin_name, pdb_path = '/tmp/complex.pdb', output_path = '/tmp'):
        self.mdin_name = mdin_name
        self.pdb_path = pdb_path
        self.output_path = output_path
        if self.output_path[-1] == '/':
            self.output_path = self.output_path[:-1]

    ''' in order to generate topology and coordinate files we need to provide LEaP configurations '''
    def create_mdin(self):
        try:
            f = open(self.mdin_name, 'w')
        except:
            raise FilePermission("Opening %s for LEaP configuration failed" % self.mdin_name)

        f.write('source oldff/leaprc.ff99SB\n')
        f.write('set default PBRadii mbondi2\n')
        f.write('source leaprc.water.tip3p\n')
        f.write('com = loadpdb %s\n' % self.pdb_path)
        f.write('saveamberparm com %s/com.prmtop %s/com.inpcrd\n' % (self.output_path, self.output_path))
        f.write('solvatebox com TIP3PBOX 12.0\n')
        f.write('saveamberparm com %s/com_solvated.prmtop %s/com_solvated.inpcrd' % (self.output_path, self.output_path))
        f.write('quit')
        f.close()

    # def run(self):
