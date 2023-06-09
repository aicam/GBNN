import subprocess
from file_parser import read_gbnsr6_output, store_frame_inpcrd
def run_gbnsr6(AMBERHOME, frame, prmtop, mdin, pre):
    inpcrd = './%s.inpcrd'%pre
    subprocess.run(['rm', inpcrd])
    subprocess.run(['rm', 'mdout'])
    store_frame_inpcrd(frame, fp=inpcrd)
    gbnsr6 = {}
    print(AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', prmtop, '-c', inpcrd, '-i', mdin)
    try:
        res = subprocess.run(
            [AMBERHOME + "/bin/gbnsr6", '-o', 'mdout', '-p', prmtop, '-c', inpcrd, '-i', mdin])
    except :
        print('Error happened in running', res)
    # if res.returncode != 0:
    #     exit(-12)
    new_res = read_gbnsr6_output('mdout')
    gbnsr6[pre + '_Etot'] = new_res['Etot']
    gbnsr6[pre + '_EKtot'] = new_res['EKtot']
    gbnsr6[pre + '_EPtot'] = new_res['EPtot']
    gbnsr6[pre + '_EELEC'] = new_res['EELEC']
    gbnsr6[pre + '_EGB'] = new_res['EGB']
    gbnsr6[pre + '_ESURF'] = new_res['ESURF']
    return gbnsr6