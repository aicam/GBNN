from exceptions import FilePermission, LEaPProgramFailed
from subprocess import Popen
from programs import find_program
class LEaP:

    def __init__(self, mdin_name = 'leap.in', tleap_prog=find_program('tleap'), pdb_path = '/tmp/complex.pdb', output_path = '/tmp'):
        self.mdin_name = mdin_name
        self.pdb_path = pdb_path
        self.output_path = output_path
        self.tleap_prog = tleap_prog
        if self.output_path[-1] == '/':
            self.output_path = self.output_path[:-1]
        self.create_mdin()

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
        f.write('saveamberparm com %s/com_solvated.prmtop %s/com_solvated.inpcrd\n' % (self.output_path, self.output_path))
        f.write('quit')
        f.close()

    def run(self):
        log = open('leap_c.log', 'w')
        process = Popen([self.tleap_prog, '-f', self.mdin_name], stdout=log)
        process_failed = bool(process.wait())

        if process_failed:
            raise LEaPProgramFailed("Generating topology and coordinate of complex failed! using tleap and %s as input" % self.mdin_name)
        log.close()