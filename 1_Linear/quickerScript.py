import shutil as sh
import os
import sys

here = os.getcwd()
dirs = os.listdir(here)

for i in dirs:
    if os.path.isdir(os.getcwd() + "/" + i):
        os.chdir(i)
        os.chdir("CCSD_T_TZ")
        # dirname="Disps_B3LYP_6-31G_2df,p_"
        dirname="Disps_CCSD_T_DZ"
        if os.path.exists(os.getcwd()+"/"+dirname):
            sh.rmtree(os.getcwd()+"/"+dirname)
        os.mkdir(dirname)
        os.chdir(dirname)
        # os.chdir("Disps_CCSD_T_DZ")
        files = os.listdir(os.getcwd())
        # for j in range(len(files)):
            # if os.path.exists(os.getcwd()+"/output."+str(j+1)+".dat"):
                # print(j)
                # os.remove("output."+str(j+1)+".dat")
        # for j in range(len(files)):
            # if os.path.exists(os.getcwd()+"/fc_int.dat"):
                # sh.move(os.getcwd()+"/fc_int.dat",os.getcwd()+"/fc_int_red.dat")
        # print(os.getcwd())
        for j in range(len(files)):
            if os.path.exists(os.getcwd()+"/templateInit.dat"):
                os.remove(os.getcwd()+"/templateInit.dat")
        # for j in range(len(files)):
        if not os.path.exists(os.getcwd()+"/templateInit.dat"):
            print(i)
            # sh.copy(here+"/templateInitDFT.dat",os.getcwd()+"/templateInit.dat")
            sh.copy(here+"/templateInitDZ.dat",os.getcwd()+"/templateInit.dat")
        # for j in range(len(files)):
            # if os.path.isdir(i):
                # print(j)
                # os.remove("output."+str(i+1)+".dat")
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
