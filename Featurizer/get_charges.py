import subprocess
import os




os.chdir("get_charges")
result = subprocess.Popen(["./get_charges.sh ~/calstate/amber/ras.pdb"], executable="/bin/bash",
                          shell=True,
                          stdout= subprocess.PIPE,
                          stderr= subprocess.PIPE)
# result = subprocess.run(['cd get_charges', ''], stdout=subprocess.PIPE)
s, e = result.communicate()
print(s.decode('utf-8'))
print(e)