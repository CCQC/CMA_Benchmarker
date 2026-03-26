import shutil as sh
import os
import sys

here = os.getcwd()
dirs = os.listdir(here)

for i in dirs:
    if os.path.isdir(os.getcwd() + "/" + i):
        os.chdir(i)
        path = os.getcwd()
        if not os.path.exists(path + "/HF_DZ"):
            sh.copytree(path + "/CCSD_T_DZ",path + "/HF_DZ")
        if not os.path.exists(path + "/HF_TZ"):
            sh.copytree(path + "/CCSD_T_DZ",path + "/HF_TZ")
        os.chdir("CCSD_T_TZ")
        path = os.getcwd()
        print(path)
        if not os.path.exists(path + "/Disps_HF_DZ"):
            os.mkdir(path + "/Disps_HF_DZ")
        if os.path.exists(path + "/Disps_CCSD_T_DZ/DispsInit") and not os.path.exists(path + "/Disps_HF_DZ/DispsInit"):
            sh.copytree(path + "/Disps_CCSD_T_DZ/DispsInit",path + "/Disps_HF_DZ/DispsInit")
        if not os.path.exists(path + "/Disps_HF_TZ"):
            os.mkdir(path + "/Disps_HF_TZ")
        if os.path.exists(path + "/Disps_MP2_TZ/DispsInit") and not os.path.exists(path + "/Disps_HF_TZ/DispsInit"):
            sh.copytree(path + "/Disps_MP2_TZ/DispsInit",path + "/Disps_HF_TZ/DispsInit")
        # sh.copy(os.getcwd() + '/' + 'zmat',os.getcwd() + '/' + 'zmat_cma1')
        os.chdir('..')
        os.chdir('..')
