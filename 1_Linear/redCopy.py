import shutil as sh
import os
import sys

here = os.getcwd()
there = '/home/vulcan/mel64643/clone_of_CMA_Benchmarker/CMA_Benchmarker/1_Closed_Shell' 
dirs = os.listdir(here)

for i in dirs:
    if os.path.isdir(os.getcwd() + "/" + i):
        os.chdir(i)
        os.chdir("CCSD_T_TZ")
        print(os.listdir(os.getcwd()))
        sh.copy(os.getcwd()+'/zmat',os.getcwd()+'/zmat_red')
        os.chdir('..')
        os.chdir("CCSD_T_DZ")
        print(os.listdir(os.getcwd()))
        sh.copy(os.getcwd()+'/zmat',os.getcwd()+'/zmat_red')
        os.chdir('..')
        os.chdir("B3LYP_6-31G_2df,p_")
        print(os.listdir(os.getcwd()))
        sh.copy(os.getcwd()+'/zmat',os.getcwd()+'/zmat_red')
        os.chdir('..')
        os.chdir('..')
