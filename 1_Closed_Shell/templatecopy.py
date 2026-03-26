import shutil as sh
import os
import sys

here = os.getcwd()
dirs = os.listdir(here)

for i in dirs:
    if os.path.isdir(os.getcwd() + "/" + i):
        os.chdir(i)
        os.chdir("CCSD_T_TZ")
        # os.chdir("Disps_CCSD_DZ")
        os.chdir("Disps_CCSD_TZ")
        # os.chdir("Disps_df_MP2_TZ")
        # os.chdir("Disps_HF_6-31G_2df,p_")
        # os.chdir("Disps_CCSD_T_DZ")
        files = os.listdir(os.getcwd())
        # for j in range(len(files)):
            # if os.path.exists(os.getcwd()+"/output."+str(j+1)+".dat"):
                # print(j)
                # os.remove("output."+str(j+1)+".dat")
        # for j in range(len(files)):
            # if os.path.exists(os.getcwd()+"/fc_int.dat"):
                # sh.move(os.getcwd()+"/fc_int.dat",os.getcwd()+"/fc_int_red.dat")
        # for j in range(len(files)):
            # if os.path.exists(os.getcwd()+"/templateInit.dat"):
                # os.remove(os.getcwd()+"/templateInit.dat")
        # sh.copy(here+"/templateInitdfMP2.dat",os.getcwd()+"/templateInit.dat")
        # for j in range(len(files)):
        if os.path.exists(os.getcwd()+"/templateInit.dat"):
            print(i)
            os.remove(os.getcwd()+"/templateInit.dat")
        if not os.path.exists(os.getcwd()+"/templateInit.dat"):
            print(i)
            # sh.copy(here+"/templateInitdfMP2.dat",os.getcwd()+"/templateInit.dat")
            # sh.copy(here+"/templateInitDFT.dat",os.getcwd()+"/templateInit.dat")
            sh.copy(here+"/templateInitDZ.dat",os.getcwd()+"/templateInit.dat")
        # for j in range(len(files)):
            # if os.path.isdir(i):
                # print(j)
                # os.remove("output."+str(i+1)+".dat")
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
    # wd = here + "/" + i
    # if os.path.isdir(wd):
        # os.chdir(i+"/CCSD_T_TZ/Disps_B3LYP_6-31G_2df,p_/")
        # files = os.listdir(os.getcwd())
        # if os.path.exists(os.getcwd()+"/old/"):
            # sh.rmtree("/old")
        # if not "/fc_int.dat" in files:
            # print(os.getcwd() + "needs fc_int.dat")
        # for j in range(len(files)):
            # if os.path.exists(os.getcwd()+"/output."+str(j+1)+".dat"):
                # os.remove("output."+str(j+1)+".dat")
        # os.chdir(here)
