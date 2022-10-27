import subprocess
from file_parser import read_gbnsr6_output, store_frame_inpcrd
def run_gbnsr6(AMBERHOME, frame, prmtop, mdin, pre):
    subprocess.run(['rm', 'complex.inpcrd'])
    subprocess.run(['rm', 'mdout'])
    inpcrd = './%s.inpcrd'%pre
    store_frame_inpcrd(frame, fp=inpcrd)
    gbnsr6 = {}
    subprocess.run(
        [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', prmtop, '-c', inpcrd, '-i', mdin])
    new_res = read_gbnsr6_output('mdout')
    gbnsr6[pre + '_Etot'] = new_res['Etot']
    gbnsr6[pre + '_EKtot'] = new_res['EKtot']
    gbnsr6[pre + '_EPtot'] = new_res['EPtot']
    gbnsr6[pre + '_EELEC'] = new_res['EELEC']
    gbnsr6[pre + '_EGB'] = new_res['EGB']
    gbnsr6[pre + '_ESURF'] = new_res['ESURF']
    return gbnsr6