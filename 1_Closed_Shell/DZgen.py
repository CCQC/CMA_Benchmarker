import shutil as sh
import os
import sys

here = os.getcwd()
dirs = os.listdir(here)

for i in dirs:
    if os.path.isdir(os.getcwd() + "/" + i):
        os.chdir(i)
        # if not os.path.exists(os.getcwd() + "/df_MP2_TZ"):
            # os.mkdir("df_MP2_TZ")
        if not os.path.exists(os.getcwd() + "/CCSD_TZ"):
            os.mkdir("CCSD_TZ")
        os.chdir("CCSD_T_TZ")
        if not os.path.exists(os.getcwd() + "/Disps_CCSD_TZ"):
            os.mkdir("Disps_CCSD_TZ")
        os.chdir('..')
        os.chdir('..')
