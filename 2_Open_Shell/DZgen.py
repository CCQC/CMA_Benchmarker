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
        if not os.path.exists(os.getcwd() + "/HF_6-31G_2df,p_"):
            os.mkdir("HF_6-31G_2df,p_")
        os.chdir("CCSD_T_TZ")
        if not os.path.exists(os.getcwd() + "/Disps_HF_6-31G_2df,p_"):
            os.mkdir("Disps_HF_6-31G_2df,p_")
        os.chdir('..')
        os.chdir('..')
