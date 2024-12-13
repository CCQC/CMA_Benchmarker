import os
import sys
import shutil
import importlib.util
from os import path
import glob
import pandas as pd
import numpy as np
from os.path import dirname, realpath
from runpy import run_path
import copy
import re
from itertools import product

from Molecule import Molecule
from SI_stuff import *

pd.set_option("display.max_columns", 15)

np.set_printoptions(precision=4)

# =======================
# Database Specifications
# =======================

# High and low levels of theory
# Available: "CCSD_T_TZ", "CCSD_T_DZ", "B3LYP_6-31G_2df,p_"
# h_theory = ["CCSD_T_aTZ"]
# h_theory = ["B3LYP_6-31G_2df,p_"]
h_theory = ["CCSD_T_TZ"]
# l_theory = ["MP2_haTZ"]
# l_theory = ["MP2_haDZ"]
# l_theory = ["MP2_aTZ"]
# l_theory = ["GFN2"]
# l_theory = ["MP2_TZ"]
# l_theory = ["MP4_DZ"]
# l_theory = ["MP4_TZ"]
# l_theory = ["MP3_TZ"]

# CMA2 Stat Theories
# l_theory = ["MP2_TZ","df_MP2_TZ","HF_TZ","B3LYP_6-31G_2df,p_","CCSD_T_DZ","CCSD_DZ","MP2_DZ","HF_DZ"]

# l_theory = ["CCSD_T_DZ","MP2_TZ"]
# l_theory = ["HF_DZ","HF_TZ"]
# l_theory = ["HF_TZ"]
# l_theory = ["MP2_DZ"]
# l_theory = ["MP2_aDZ"]
# l_theory = ["MP2_aDZ"]
# l_theory = ["MP2_TZ","CCSD_T_DZ","B3LYP_6-31G_2df,p_"]
# l_theory = ["df_MP2_TZ"]
l_theory = ["MP2_TZ"]
# l_theory = ["CCSD_T_haTZ"]
# l_theory = ["CCSD_T_haDZ"]
# l_theory = ["CCSD_T_aDZ"]
# l_theory = ["CCSD_T_aTZ"]
# l_theory = ["CCSD_haTZ"]
# l_theory = ["CCD_TZ"]
# l_theory = ["CCSD_DZ"]
# l_theory = ["CCSD_T_TZ"]
# l_theory = ["HF_DZ"]
# l_theory = ["CCSD_T_DZ"]
# l_theory = ["CCSD_TZ"]
# l_theory = ["HF_6-31G_2df,p_"]
# l_theory = ["B3LYP_6-31G_2df,p_"]
# l_theory = [
# # "CCSD_aTZ",
# # "CCSD_haTZ",
# "CCSD_T_aDZ",
# "CCSD_T_DZ",
# "CCSD_T_haDZ",
# "CCSD_T_haTZ",
# "CCSD_T_TZ",
# "MP2_aDZ",
# "MP2_aTZ",
# "MP2_DZ",
# "MP2_haDZ",
# "MP2_haTZ",
# "MP2_TZ"]

# l_theory = ["CCSD_T_DZ", "B3LYP_6-31G_2df,p_"]
combos = list(product(h_theory,l_theory))

# cma1_energy_regexes = ["!CCD\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["CCSD\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["\sMP2\s*t?o?t?a?l? energy:\s+(\-\d+\.\d+)"]

# CMA2 Stat Theories
# cma1_energy_regexes = ["RMP2 energy\s+(\-\d+\.\d+)","Grab this energy\s+(\-\d+\.\d+)","a","Grab this energy\s+(\-\d+\.\d+)","a","a","a","!RHF STATE 1.\d+ Energy\s+(\-\d+\.\d+)"]

# cma1_energy_regexes = ["!CCSD\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["!RHF-UCCSD\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["CCSD\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["\s+MP2\s*t?o?t?a?l?\s*energy:\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["MP4\(SDTQ\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["MP3\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["!RMP2 STATE 1.\d+ Energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["!UMP2 STATE 1.\d+ Energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)","!RMP2 STATE 1.\d+ Energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["!RHF STATE 1.\d+ Energy\s+(\-\d+\.\d+)","!RHF STATE 1.\d+ Energy\s+(\-\d+\.\d+)"]
cma1_energy_regexes = ["!RHF STATE 1.\d+ Energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)","MP2\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["Grab this energy\s+(\-\d+\.\d+)"]
# cma1_energy_regexes = ["Grab this energy\s+(\-\d+\.\d+)","",""]
# cma1_energy_regexes = [
# # "CCSD_aTZ",
# # "CCSD_haTZ",
# "CCSD_T_aDZ",
# "CCSD_T_DZ",
# "CCSD_T_haDZ",
# "CCSD_T_haTZ",
# "CCSD_T_TZ",
# "MP2_aDZ",
# "MP2_aTZ",



# cma1_energy_regexes = ["CCSD\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
cma1_gradient_regex = []
# cma1_gradient_regex = ["\s*virial=\S\S+\.\d+E\S\d+\S\>\s+"]
# cma1_energy_regexes = ["\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)","Grab this energy (\-\d+\.\d+)"]
# cma1_energy_regexes = ["\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)",[r"Total Gradient",r"tstop"]]
# cma1_success_regexes = ["Variable memory released","beer"]
# cma1_success_regexes = ["Molpro calculation terminated"]
# cma1_success_regexes = ["!MP2\s*t?o?t?a?l?\s*energy\s+\-\d+\.\d+"]
# cma1_success_regexes = ["total\s*energy\s+\-\d+\.\d+"]
# cma1_success_regexes = ["Variable memory released","Variable memory released"]
cma1_success_regexes = ["Variable memory released"]

# CMA2 Stat Theories
# cma1_success_regexes = ["Variable memory released","beer","a","beer","a","a","a","Variable memory released"]

# cma1_success_regexes = ["beer","",""]
# cma1_success_regexes = ["beer"]
# cma1_success_regexes = [
# # "CCSD_aTZ",
# # "CCSD_haTZ",
# "CCSD_T_aDZ",
# "CCSD_T_DZ",
# "CCSD_T_haDZ",
# "CCSD_T_haTZ",
# "CCSD_T_TZ",
# "MP2_aDZ",
# "MP2_aTZ",
# "MP2_DZ",
# "MP2_haDZ",
# "MP2_haTZ",
# "MP2_TZ"]

# Coordinates types to use
# Available: "Nattys", "Redundant", "ZMAT" (not yet tho)
# coord_type = ["Nattys", "Redundant"]
# coord_type = ["Redundant"]
coord_type = ["Nattys"]

# Specify paths to grab data from
# Options: '/1_Closed_Shell', '/1_Linear', '/1*', '/2_Open_Shell', '/2_Linear', '/2*'
# paths = ['/2_Open_Shell']
# paths = ['/4_G3_99']
# paths = ['/3_Dimers']
# paths = ['/1_Closed_Shell','/2_Open_Shell']

# paths = ['/1*','/2*']
# paths = ['/2*']
# job_list = ["3.4"]
# job_list = ["4.10"]
# job_list = ["2.18"]
job_list = ["1.91"]
# job_list = ["0.2"]
# exclude_list = ["1.91","1.57"]
# exclude_list = ["1.91","1.57","2.14"]
# exclude_list = ["1.91"]
exclude_list = []


# Various output control statements
# n = 1                    # Number of CMA2 corrections (n = 0 -> CMA0)
n = 0                    # Number of CMA2 corrections (n = 0 -> CMA0)
# n = 36                    # Number of CMA2 corrections (n = 0 -> CMA0)
# n = 15                    # Number of CMA2 corrections (n = 0 -> CMA0)
# n = 3                    # Number of CMA2 corrections (n = 0 -> CMA0)
# xi_tol = [100.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.2,0.18,0.16,0.14,0.12,0.10,0.08,0.075,0.07,0.065,0.06,0.055,0.05,0.045,0.04,0.036,0.032,0.028,0.024,0.02,0.018,0.016,0.014,0.012,0.011,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0.0]    # Xi value for cutoff in determining CMA2 off diags
# xi_tol = [100,0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06,0.04,0.02,0.01,0.005]    # Xi value for cutoff in determining CMA2 off diags
# xi_tol = [10.0, 0.5, 0.2, 0.1, 0.05, 0.04, 0.033, 0.028, 0.0245, 0.021, 0.019, 0.017, 0.01, 0.005, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]    # Xi values for MP2/TZ fig
# xi_tol = [10.0, 0.5, 0.2, 0.1, 0.047, 0.036, 0.03, 0.025, 0.021, 0.019, 0.017, 0.01, 0.0075, 0.0062, 0.005, 0.0035, 0.0025, 0.0017, 0.001, 0.00075, 0.0005, 0.00025, 0.0001, 7.5e-05, 5e-05, 2.5e-05, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]    # Xi values for (T)/DZ fig
# xi_tol = [10.0, 3.0, 2.0, 0.20, 0.07, 0.04, 0.03, 0.025, 0.022, 0.018, 0.01, 0.005, 0.004, 0.003, 0.0002, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]    # Xi values for B3LYP/6-31G(2df,p) fig
# xi_tol = []    # Xi value for cutoff in determining CMA2 off diags
xi_tol = [0.04]    # Xi value for cutoff in determining CMA2 off diags
od_inds = [[16,17]]         # Contains a list of lists, where the sublists contain off-diagonal elements to be computed in CMA-1
# cmaA = False             # Run CMA1 instead of CMA0
cmaA = True             # Run CMA1 instead of CMA0
csv = False               # Generate database .csv file
# csv = True               # Generate database .csv file
SI = False                # Generate LaTeX SI file
# SI = True               # Generate LaTeX SI file
compute_all = False       # run calculations for all or a select few
# compute_all = True       # run calculations for all or a select few
# off_diag_bands = False   # (CMA2/3 ONLY) If set to true, "n" off-diag bands selected, if false, "n" largest fc will be selected
                           # off_diag_bands is now an obsolete option
# off_diag = 0   # Set this option for CMA0
# off_diag = 1   # Set this option for CMA1. Additional off-diagonal elements will need to be specified using ___.
off_diag = 2   # Set this option for CMA2. Off-diags will be auto generated, but an aux hessian will need be specified using ___.
deriv_level = 0         # (CMA1) if 0, compute initial hessian by singlepoints. If 1, compute initial hessian with findif of gradients
# second_order = True    # If True, read in cartesian gradient and force constant info to be converted to internal coordinates.
second_order = False    # If False, generate displacements to manually compute the CMA-0A internal coord force constants.
# coord_type_init = "cartesian" # Toggle this for type of coordinate used in inital force constant computations
coord_type_init = "internal" # Toggle this for type of coordinate used in inital force constant computations

# =====================
# Some useful functions
# =====================

def freq_diff(F1, F2):
    return F2 - F1

def averages(F_diff):
    return np.average(F_diff), np.average(abs(F_diff)) 

def stdev(F_diff):
    return np.std(F_diff)

def highlight_greaterthan(x):
    print('this is x.Ref__Redundant')
    print(x.Ref_Redundant) 
    #return np.where((x.F2diff > 0.5), props, '')
    if abs(x.Ref_Redundant) > 0.5:
        print('oi!!!!') 
        return ['background-color:yellow']
    else:
         return ['background-color:white']

def build_dataframe(d,basename):
    df = pd.DataFrame(data=d)
    df_i = df.to_csv('out.csv',index=False)
    # adds a space in between each molecule. 
    df.loc[df.shape[0]] = [None, None, None, None, None, None, None] 
    df.loc[df.shape[0]] = [None, None, None, None, None, None, None] 
    df.loc[df.shape[0]] = [None, None, None, None, None, None, None] 

    #df.loc[df.shape[0]] = [None, None, 'Signed Avg.', averages(F1diff)[0], 'Signed Avg.', averages(F1diff)[0], averages(F1F2diff)[0]]
    #df.loc[df.shape[0]] = [None, None, 'Average   .', averages(F1diff)[1], 'Average    ', averages(F1diff)[1], averages(F1F2diff)[1]]
    #df.loc[df.shape[0]] = [None, None, 'Std. Dev  .', stdev(F1diff),    'Std. Dev.  ', stdev(F1diff), stdev(F1F2diff)]

    df.loc[0,('Molecule')] = str(basename) 
    #df_i: dataframe, individual for each child directory
    frame.append(df)
    return frame, df_i 

def return_path_base(jobpath):
    name = os.path.basename(os.path.normpath(jobpath)) 
    return name

# ====================================
# Print out information about database
# ====================================

hq = os.getcwd()
jobb_list = []
path_ind = []

print("CMA Database Generation\n")
print(
"""
      CCCCCCCC      MMMMMM     MMMMMM           AAAAA
    CCCCCCCCCCCC    MMMMMMM   MMMMMMM          AAAAAAA
   CCCC      CCCC   MMMMMMMM MMMMMMMM         AAAA AAAA
  CCCC              MMMM MMMMMMM MMMM        AAAA   AAAA
  CCCC              MMMM  MMMM   MMMM       AAAA     AAAA
  CCCC              MMMM         MMMM      AAAA       AAAA
  CCCC              MMMM         MMMM     AAAAAAAAAAAAAAAAA 
   CCCC      CCCC   MMMM         MMMM    AAAAAAAAAAAAAAAAAAA
    CCCCCCCCCCCC    MMMM         MMMM   AAAA             AAAA
      CCCCCCCC      MMMM         MMMM  AAAA               AAAA
"""
)



print("Contributors: Dr. Mitchell Lahm, Dr. Nathaniel Kitzmiller, Dr. Henry Mull, Laura Olive, Jace Jin")
print()
print("Combinations of levels of theory (high, low): ", end="")
print(*combos, sep=", ")
print("Coordinate types: ", end="")
print(*coord_type, sep=", ")
print(f"Number of off-diagonal corrections: {n}")
print()

# Grab jobs from each path and order them numerically by ID
if compute_all:
    for path in paths:
        path_ind.append(len(jobb_list))
        tmp_list = glob.glob(hq + path + "/[1-9]*_*/")
        ind = np.argsort(np.array([int(re.search(r"/\d_.*/(\d*)_.*", name).group(1)) for name in tmp_list]))
        tmp_list = [tmp_list[i] for i in ind]
        jobb_list += tmp_list
    if len(exclude_list):
        excludee_list = []
        for job in exclude_list:
            id1, id2 = job.split(".")
            excludee_list += glob.glob(hq + f"/{id1}_*" + f"/{id2}_*/")
        for path in excludee_list:
            jobb_list.remove(path)

else:
    for job in job_list:
        id1, id2 = job.split(".")
        jobb_list += glob.glob(hq + f"/{id1}_*" + f"/{id2}_*/")

# Gives printout for each molecule in jobb_list
print(f"Generating database entries for {len(jobb_list)} jobs:",end="")
count = 0
for i, job in enumerate(jobb_list):
    if i in path_ind:
        print("\n\nDirectory: {}".format(paths[path_ind.index(i)]),end="")
        count = 0
    if count % 5 == 0:
        print("\n{0:20}".format(return_path_base(job)),end="")
        count += 1
    else:
        print("{0:20}".format(return_path_base(job)),end="")
        count += 1
print("\n")

# Initialize frames for pandas database
frame = []
framez = []
framem = []
if n > 0:
    frame2 = []
    frame2e = []
    frame2m = []

# Start SI file or clears contents
if SI:
    si = open("SI.tex", "w")
    si.write(header)
section = ""

# ============
# Do the thing
# ============

def execute():
    for i, job in enumerate(jobb_list):
        os.chdir(job)
        sys.path.insert(0,job)
        options = None

        # Initialize objects
        mol = Molecule(job,h_theory)
        basename = return_path_base(job) 
        d = {'Molecule' : None}     # Ensures molecule is first column
        z = {'Molecule' : None}     # Ensures molecule is first column
        m = {'Molecule' : None}     # Ensures molecule is first column
        if off_diag > 0: 
        # if n > 0: 
            d2 = {'Molecule' : None}
            d2e = {'Molecule' : None}
            d2m = {'Molecule' : None}
            # z2 = {'Molecule' : None}

        if i in path_ind:
            print(f"Currently running jobs in {paths[path_ind.index(i)]}\n")
        
        print("////////////////////////////////////////////")
        print(f"//{basename:^40}//")
        print("////////////////////////////////////////////")
        countt = 0
        # Run CMA for each combination of theory
        for combo in combos:
            # Grab geometry information 
            mol.get_geoms(combo)
            if not mol.direc_complete:
                break
 
            if not cmaA:
                # Copy the necessary files with correct names
                
                # Run for each coord type
                for coord in coord_type:
                
                    # Kept in for its historical significance
                    if coord == "Nattys":
                        print('ReeeeEEEEEeEEEEEEEEEEEEeEeeeeeeeeeeeeeeeeeEEEE')
                    elif coord == "Redundant":
                        print("Catalina wine mixer " + str(i))
                
                    print()
                    print("="*50)
                    print(" "*16+"Current Parameters")
                    print("-"*50)
                    print(f"  Job                     {mol.name} ({mol.ID})") 
                    print(f"  High level of theory    {combo[0]}")
                    print(f"  Low level of theory     {combo[1]}")
                    print(f"  Coordinate type         {coord}")
                    print("="*50)
                    print()
                
                    from Merger import Merger
                    execMerger = Merger()
                     
                    #Specify options for Merger
                    sym_sort = np.array([])
                    if coord == "Nattys":
                        shutil.copyfile(job + combo[1] + "/zmat", job + "zmat")
                        shutil.copyfile(job + combo[1] + "/fc.dat", job + "fc.dat")
                        shutil.copyfile(job + combo[0] + "/zmat", job + "zmat2")
                        shutil.copyfile(job + combo[0] + "/fc.dat", job + "fc2.dat")       
                        execMerger.options.man_proj = True
                        execMerger.options.coords = 'Custom'
                
                        #import the manual_projection module from the specific molecule directory 
                        spec = importlib.util.spec_from_file_location("manual_projection",  job + "/manual_projection.py")
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        project_obj = foo.Projection(None)
                        project_obj.run()
                        Proj = copy.copy(project_obj.Proj)
                        try:
                            sym_sort = copy.copy(project_obj.sym_sort)
                        except:
                            pass
                        mol.proj = Proj
                        mol.get_nattys(combo)
                    
                    else:
                        shutil.copyfile(job + combo[1] + "/zmat_red", job + "zmat")
                        shutil.copyfile(job + combo[1] + "/fc.dat", job + "fc.dat")
                        shutil.copyfile(job + combo[0] + "/fc.dat", job + "fc2.dat")       
                        execMerger.options.man_proj = False
                        execMerger.options.coords = coord
                        Proj = None
                        if 'Linear' in job:
                            shutil.copyfile(job + combo[0] + "/zmat_cma1", job + "zmat2")
                            execMerger.options.coords = 'Custom'
                        else:
                            shutil.copyfile(job + combo[0] + "/zmat", job + "zmat2")
                            
                
                    execMerger.options.n_cma2 = n
                    execMerger.options.off_diag = off_diag
                    # execMerger.options.off_diag = off_diag_bands
                
                    # Run CMA
                    execMerger.run(execMerger.options, Proj, sym_sort=sym_sort, coord_type_init=coord_type_init)
                
                    # Collect data
                    if coord_type.index(coord) == 0:
                        d[f'Ref ({combo[0]})'] = execMerger.reference_freq
                        z[f'Ref ({combo[0]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133)
                        mol.freqs[f'Ref ({combo[0]})'] = execMerger.reference_freq
                        d[f'Ref ({combo[1]})'] = execMerger.ref_init
                        z[f'Ref ({combo[1]})'] = np.sum(execMerger.ref_init)/(2*349.7550881133)
                        mol.freqs[f'Ref ({combo[1]})'] = execMerger.ref_init
                        

                        # Number the modes
                        d['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                        z['Molecule'] = [f"{mol.name} ({mol.ID})"]
                
                    if coord == "Nattys":
                        d[f'Natty ({combo[1]})'] = execMerger.Freq_CMA0
                        z[f'Natty ({combo[1]})'] = np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                        d[f'Ref - Nat ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_CMA0)
                        z[f'Ref - Nat ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                        mol.freqs[f'Natty ({combo[1]})'] = execMerger.Freq_CMA0
                    if coord == "Redundant":
                        if 'Linear' not in job:
                            d[f'Red ({combo[1]})'] = execMerger.Freq_redundant
                            z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_redundant)/(2*349.7550881133)
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_redundant)
                            z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_redundant)/(2*349.7550881133)
                            mol.freqs[f'Red ({combo[1]})'] = execMerger.Freq_redundant
                        else:
                            d[f'Red ({combo[1]})'] = execMerger.Freq_CMA0
                            z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_CMA0)
                            z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                            mol.freqs[f'Red ({combo[1]})'] = execMerger.Freq_CMA0
                
                    # Collect data for CMA2
                    if n > 0:
                        if coord == "Nattys":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                            d2[f"Ref {combo[0]}"] = execMerger.reference_freq
                            d2[f'Natty ({combo[1]})'] = execMerger.Freq_CMA0
                            cma2_freqs_natty = execMerger.Freq_cma2 
                          
                            d2[f'Natty CMA2 ({combo[1]})'] = cma2_freqs_natty 
                            d2[f'Ref - Natty ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_CMA0)
                            d2[f'Ref - Natty CMA2 ({combo[1]})'] = freq_diff(execMerger.reference_freq, cma2_freqs_natty)
                            print('Give us CMA2!')                   
                        if coord == "Redundant":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                            d2[f"Ref {combo[0]}"] = execMerger.reference_freq
                            if 'Linear' not in job:
                                d2[f'Red ({combo[1]})'] = execMerger.Freq_redundant
                            else:
                                d2[f'Red ({combo[1]})'] = execMerger.Freq_CMA0
                            cma2_freqs_red = execMerger.Freq_cma2 
                          
                            d2[f'Natty CMA2 ({combo[1]})'] = cma2_freqs_red 
                            if 'Linear' not in job:
                                d2[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_redundant)
                            else:
                                d2[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_CMA0)
                            d2[f'Ref - Red CMA2 ({combo[1]})'] = freq_diff(execMerger.reference_freq, cma2_freqs_red)
                            print('Give us CMA2!')                   
                
                    # delete objects so they are forced to reload
                    del execMerger
                    del Merger
                    if coord == "Nattys":
                        del project_obj
                
                # end of coord loop

                # Difference between Natty and Redundant freqs
                if 'Nattys' in coord_type and 'Redundant' in coord_type:
                    d[f'Nat - Red {combo[1]}'] = freq_diff(d[f'Natty ({combo[1]})'], d[f'Red ({combo[1]})'])
                
            #======================
            # Mitchell's playground
            #======================
            elif not mol.direc_complete:
                continue
            
            elif cmaA:
                # move into directory with higher level geom, fc.dat, and Disp directories
                os.chdir(f"{job}/")
                print(f"I am in {os.getcwd()} and I can see {os.listdir()}")
                print(job + combo[0]) 
                
                for coord in coord_type:
                    print()
                    print("="*50)
                    print(" "*16+"Current Parameters")
                    print("-"*50)
                    print(f"  Job                     {mol.name} ({mol.ID})") 
                    print(f"  High level of theory    {combo[0]}")
                    print(f"  Low level of theory     {combo[1]}")
                    print(f"  Coordinate type         {coord}")
                    print("="*50)
                    print()
                    from Merger import Merger
                    execMerger = Merger(cma1_path= "/" + combo[0]+"/Disps_" + combo[1])
                    if os.path.exists(os.getcwd() + "/" + combo[0]+"/Disps_" + combo[1] + "/templateInit.dat"):
                        #change to True if you need the displacements generated
                        execMerger.options.calc_init = True
                    # execMerger.options.calc_init = False

                    if os.path.exists(os.getcwd() + "/" + combo[0]+"/Disps_" + combo[1] + "/DispsInit"):
                        execMerger.options.calc_init = False
                        execMerger.options.gen_disps_init = False
                    execMerger.options.cart_insert_init = 9
                    if combo[1] == "B3LYP_6-31G_2df,p_":
                        execMerger.options.other_F_matrix = 'HF_6-31G_2df,p_'
                    elif combo[1] == "CCSD_T_DZ":
                        execMerger.options.other_F_matrix = 'HF_DZ'
                    elif combo[1] == "MP2_TZ":
                        execMerger.options.other_F_matrix = 'HF_TZ'
                    else:
                        execMerger.options.other_F_matrix = ''
                    # execMerger.options.other_F_matrix = 'MP2_TZ'
                    if len(execMerger.options.other_F_matrix):
                        if os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_int_nat.dat"):
                            shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_int_nat.dat",os.getcwd()+"/inter_fc.dat")
                    if combo[1] == "CCSD_T_DZ":
                        # execMerger.options.cart_insert_init = 24
                        execMerger.options.cart_insert_init = 9
                    elif combo[1] == "B3LYP_6-31G_2df,p_" or combo[1] == "HF_6-31G_2df,p_" or combo[1] == "df_MP2_TZ":
                        execMerger.options.cart_insert_init = 4
                        execMerger.options.program_init = "psi4@master"
                    execMerger.options.coords = coord
                    execMerger.options.n_cma2 = n
                    # execMerger.options.off_diag = off_diag_bands
                    execMerger.options.off_diag = off_diag
                    execMerger.options.deriv_level = deriv_level
                    execMerger.options.second_order = second_order
                    sym_sort = np.array([])
                    if coord == "Nattys":
                        if second_order:
                            try:
                                shutil.copyfile(job + combo[0] + "/Disps_" + combo[1] + "/fc_cart.dat", job + "fc.dat")
                                shutil.copyfile(job + combo[0] + "/Disps_" + combo[1] + "/fc_cart.grad", job + "fc.grad")
                            except:
                                print('Once again, the directory does not contain the sufficient files for the specified job')
                                # mol.direc_complete = False
                                # break 
                        try: 
                            shutil.copyfile(job + combo[0] + "/zmat", job + "zmat")
                            shutil.copyfile(job + combo[0] + "/zmat", job + "zmat2")
                            shutil.copyfile(job + combo[0] + "/fc.dat", job + "fc2.dat")      
                        except:
                            print('Once again, the directory does not contain the sufficient files for the specified job')
                            mol.direc_complete = False
                            break 
                        cma1_coord = "nat"
                        execMerger.options.man_proj = True
                        execMerger.options.coords = 'Custom'
                        spec = importlib.util.spec_from_file_location("manual_projection",  job + "/manual_projection.py")
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        project_obj = foo.Projection(None)
                        project_obj.run()
                        Proj = copy.copy(project_obj.Proj)
                        mol.proj = Proj
                        np.set_printoptions(precision=4,threshold=sys.maxsize,linewidth=500)
                        try:
                            sym_sort = copy.copy(project_obj.sym_sort)
                        except:
                            pass
                        mol.get_nattys(combo)
                
                    else:
                        if second_order:
                            try:
                                shutil.copyfile(job + combo[0] + "/Disps_" + combo[1] + "/fc_cart.dat", job + "fc.dat")
                                shutil.copyfile(job + combo[0] + "/Disps_" + combo[1] + "/fc_cart.grad", job + "fc.grad")
                            except:
                                print('Once again, the directory does not contain the sufficient files for the specified job')
                                mol.direc_complete = False
                                break 
                        try: 
                            shutil.copyfile(job + combo[0] + "/zmat_red", job + "zmat")
                            shutil.copyfile(job + combo[0] + "/zmat_red", job + "zmat2")
                            shutil.copyfile(job + combo[0] + "/fc.dat", job + "fc2.dat")       
                            #shutil.copyfile(job + combo[0] + "/zmat_cma1", job + "zmat")
                            #shutil.copyfile(job + combo[0] + "/zmat_cma1_Final", job + "zmat2")
                        except:
                            print('Once again, the directory does not contain the sufficient files for the specified job')
                            mol.direc_complete = False
                            break 
                        cma1_coord = "red"
                        execMerger.options.man_proj = False
                        execMerger.options.coords = coord
                        execMerger.options.gradient_regex = cma1_gradient_regex
                        Proj = None
                        if 'Linear' in job:
                            execMerger.options.coords = 'Custom'
                    execMerger.run(execMerger.options,Proj,energy_regex=cma1_energy_regexes[countt],success_regex=cma1_success_regexes[countt],cma1_coord=cma1_coord, sym_sort=sym_sort, xi_tol=xi_tol, coord_type_init=coord_type_init, od_inds=od_inds)
                    # Collect the data in dictionary d to add it to the database
                    # e.g. d[f"Ref {combo[0]}"] = execMerger.reference_freq
                    # d[f"CMA1 {combo[1]}"] = execMerger.Freq_redundant
                    
                    ref_freq = execMerger.reference_freq.copy()
                    freq_indices = [i for i in range(len(ref_freq))]
                    freq_indices = np.array(freq_indices)
                    
                    # Collect data
                    if coord_type.index(coord) == 0:
                    # if coord_type.index(coord) == 1:
                        ref_freq = execMerger.reference_freq.copy()
                        ref_freq_init = execMerger.ref_init.copy()
                        
                        # raise RuntimeError
                        d[f'Ref ({combo[0]})'] = ref_freq
                        z[f'Ref ({combo[0]})'] = np.sum(execMerger.reference_freq)/2
                        # z[f'Ref ({combo[0]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133)
                        mol.freqs[f'Ref ({combo[0]})'] = ref_freq
                        
                        d[f'Ref ({combo[1]})'] = ref_freq_init
                        d[f'Pure - Ref ({combo[1]})'] = freq_diff(ref_freq, ref_freq_init)
                        d[f'ABS Pure - Ref ({combo[1]})'] = np.abs(freq_diff(ref_freq, ref_freq_init))
                        z[f'Ref ({combo[1]})'] = np.sum(ref_freq_init)/2
                        z[f'Pure - Ref ({combo[1]})'] = np.sum(ref_freq_init)/2 - np.sum(ref_freq)/2
                        # z[f'Ref ({combo[1]})'] = np.sum(ref_freq_init)/(2*349.7550881133)
                        m[f'Pure - Ref ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, ref_freq_init)))
                        mol.freqs[f'Initial ({combo[1]})'] = ref_freq_init
                        
                        # Number the modes
                        d['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                        z['Molecule'] = [f"{mol.name} ({mol.ID})"]
                    
                    if coord == "Nattys":
                        d['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in freq_indices]
                        z['Molecule'] = [f"{mol.name} ({mol.ID})"]
                        m['Molecule'] = [f"{mol.name} ({mol.ID})"]
                        custom_freq = execMerger.Freq_CMA0.copy()
                        d[f'Natty ({combo[1]})'] = custom_freq
                        # d[f'Natty ({combo[1]})'] = execMerger.Freq_custom
                        z[f'Natty ({combo[1]})'] = np.sum(custom_freq)/2
                        # z[f'Natty ({combo[1]})'] = np.sum(execMerger.Freq_custom)/2
                        # z[f'Natty ({combo[1]})'] = np.sum(execMerger.Freq_custom)/(2*349.7550881133)
                        d[f'Ref - Nat ({combo[1]})'] = freq_diff(ref_freq, custom_freq)
                        d[f'ABS Ref - Nat ({combo[1]})'] = np.abs(freq_diff(ref_freq, custom_freq))
                        # d[f'Ref - Nat ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_custom)
                        z[f'Ref - Nat ({combo[1]})'] = np.sum(custom_freq)/2 - np.sum(ref_freq)/2
                        # z[f'Ref - Nat ({combo[1]})'] = np.sum(execMerger.reference_freq)/2 - np.sum(execMerger.Freq_custom)/2
                        # z[f'Ref - Nat ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_custom)/(2*349.7550881133)
                        m[f'Ref - Nat ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, custom_freq)))
                        mol.freqs[f'Natty ({combo[1]})'] = custom_freq
                        # mol.freqs[f'Natty ({combo[1]})'] = execMerger.Freq_custom
                    if coord == "Redundant":
                        if 'Linear' not in job:
                            red_freq = execMerger.Freq_redundant.copy()

                            d[f'Red ({combo[1]})'] = red_freq
                            # d[f'Red ({combo[1]})'] = execMerger.Freq_redundant
                            z[f'Red ({combo[1]})'] = np.sum(red_freq)/2
                            # z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_redundant)/2
                            # z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_redundant)/(2*349.7550881133)
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, red_freq)
                            d[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, red_freq))
                            # d[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_redundant)
                            z[f'Ref - Red ({combo[1]})'] = np.sum(ref_freq)/2 - np.sum(red_freq)/2
                            # z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/2 - np.sum(execMerger.Freq_redundant)/2
                            # z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_redundant)/(2*349.7550881133)
                            m[f'Ref - Red ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, red_freq)))
                            # m[f'Ref - Red ({combo[1]})'] = np.max(np.abs(freq_diff(execMerger.reference_freq, execMerger.Freq_custom)))
                            mol.freqs[f'Red ({combo[1]})'] = red_freq
                            # mol.freqs[f'Red ({combo[1]})'] = execMerger.Freq_redundant
                        else:
                            cust_freq = execMerger.Freq_CMA0.copy()
                            d[f'Red ({combo[1]})'] = cust_freq
                            # d[f'Red ({combo[1]})'] = execMerger.Freq_custom
                            # z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_custom)/(2*349.7550881133)
                            z[f'Red ({combo[1]})'] = np.sum(cust_freq)/2
                            # z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_custom)/2
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, cust_freq)
                            d[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, cust_freq))
                            # d[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_custom)
                            z[f'Ref - Red ({combo[1]})'] = np.sum(ref_freq)/2 - np.sum(cust_freq)/2
                            # z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/2 - np.sum(execMerger.Freq_custom)/2
                            # z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_custom)/(2*349.7550881133)
                            m[f'Ref - Red ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, cust_freq)))
                            # m[f'Ref - Red ({combo[1]})'] = np.max(np.abs(freq_diff(execMerger.reference_freq, execMerger.Freq_custom)))
                            mol.freqs[f'Red ({combo[1]})'] = cust_freq
                            # mol.freqs[f'Red ({combo[1]})'] = execMerger.Freq_custom
                    
                    #Collect data for CMA2
                    if off_diag > 0:
                    # if n > 0:

                        if coord == "Nattys":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(ref_freq))]
                            d2e['Molecule'] = [f"{mol.name} ({mol.ID})"]
                            d2m['Molecule'] = [f"{mol.name} ({mol.ID})"]
                            # d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                            d2[f"Ref {combo[0]}"] = ref_freq
                            # d2[f"Ref {combo[0]}"] = execMerger.reference_freq
                            d2[f'Natty ({combo[1]})'] = custom_freq
                            # d2[f'Natty ({combo[1]})'] = execMerger.Freq_custom
                            # cma2_freqs_natty = execMerger.Freq_cma2.copy()
                            if off_diag == 1:
                                cma1_freqs_natty = execMerger.Freq_cma1.copy()
                                mol.freqs[f'Natty CMA1 ({combo[1]})'] = cma1_freqs_natty
                                d2[f'Natty CMA1 ({combo[1]})'] = cma1_freqs_natty
                                # d2[f'Ref - Natty ({combo[1]})'] = freq_diff(ref_freq, custom_freq)
                                # d2[f'Ref - Natty ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_custom)
                                d2[f'Ref - Natty CMA1 ({combo[1]})'] = freq_diff(ref_freq, cma1_freqs_natty)
                                d2[f'ABS Ref - Natty CMA1 ({combo[1]})'] = np.abs(freq_diff(ref_freq, cma1_freqs_natty))
                                d2m[f'Ref - Natty CMA1 ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, cma1_freqs_natty)))
                            elif off_diag == 2:
                                cma2_freqs_natty = execMerger.Freq_cma2.copy()
                                for i in range(len(xi_tol)):
                                    mol.freqs[f'Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = cma2_freqs_natty[i]
                                    d2[f'Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = cma2_freqs_natty[i] 
                                    # d2[f'Ref - Natty ({combo[1]})'] = freq_diff(ref_freq, custom_freq)
                                    # d2[f'Ref - Natty ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_custom)
                                    d2[f'Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = freq_diff(ref_freq, cma2_freqs_natty[i])
                                    d2[f'ABS Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = np.abs(freq_diff(ref_freq, cma2_freqs_natty[i]))
                                    d2e[f'Natty CMA2 eta_num ({combo[1]}) xi ({xi_tol[i]})'] = execMerger.eta_num[i]
                                    d2e[f'Natty CMA2 eta_denom ({combo[1]}) xi ({xi_tol[i]})'] = execMerger.eta_denom[i]
                                    d2e[f'Natty CMA2 tot_off_diags ({combo[1]}) xi ({xi_tol[i]})'] = execMerger.total_off_diags[i]
                                    d2m[f'Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = np.max(np.abs(freq_diff(ref_freq, cma2_freqs_natty[i])))
                                # d2[f'Ref - Natty CMA2 ({combo[1]})'] = freq_diff(execMerger.reference_freq, cma2_freqs_natty)
                        # raise RuntimeError
                        if coord == "Redundant":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(ref_freq))]
                            d2e['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(ref_freq))]
                            # d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                            d2[f"Ref {combo[0]}"] = ref_freq
                            # d2[f"Ref {combo[0]}"] = execMerger.reference_freq
                            if 'Linear' not in job:
                                d2[f'Red ({combo[1]})'] = red_freq
                                # d2[f'Red ({combo[1]})'] = execMerger.Freq_redundant
                                d2[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, red_freq)
                                d2[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, red_freq))
                                # d2[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_redundant)
                            else:
                                # d2[f'Red ({combo[1]})'] = execMerger.Freq_custom
                                d2[f'Red ({combo[1]})'] = cust_freq
                                d2[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, cust_freq)
                                d2[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, cust_freq))
                                # d2[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_custom)
                            cma2_freqs_red = execMerger.Freq_cma2 
                          
                            d2[f'Natty CMA2 ({combo[1]})'] = cma2_freqs_red 
                            d2[f'Ref - Red CMA2 ({combo[1]})'] = freq_diff(ref_freq, cma2_freqs_red)
                            d2[f'ABS Ref - Red CMA2 ({combo[1]})'] = np.abs(freq_diff(ref_freq, cma2_freqs_red))
                            # d2[f'Ref - Red CMA2 ({combo[1]})'] = freq_diff(execMerger.reference_freq, cma2_freqs_red)


                    del execMerger
                    del Merger
            if 'Nattys' in coord_type and 'Redundant' in coord_type:
                d[f'Nat - Red {combo[1]}'] = freq_diff(d[f'Natty ({combo[1]})'], d[f'Red ({combo[1]})'])
            countt += 1

        # end of combo loop
        # print("begin:")
        if mol.direc_complete: 
            # Print molecule information
            print("begin:")
            if SI:
                si.write(mol.build_latex_output(cma1=cma1,combos=combos,xi_tol=xi_tol,sym_sort=sym_sort))
            
            # Clean up job directory
            if not cmaA:
                os.remove("fc.dat")
            try: 
                os.remove("zmat")
                os.remove("zmat2")
                if second_order:
                    os.remove("fc.dat")
                    os.remove("fc.grad")
                os.remove("fc2.dat")
                os.remove("inter_fc.dat")
            except:
                print('These are not the files you are looking for') 
            
            if coord_type[0] == 'Nattys':
                print("end:")
                mol.run()
            sys.path.remove(job)
            del mol


            # Add to pandas dataframe
            if csv:
                df = pd.DataFrame(data=d)
                zf = pd.DataFrame(data=z)
                mf = pd.DataFrame(data=m)
                frame.append(df)
                framez.append(zf)
                framem.append(mf)
            
            if n > 0:
                # d2 ={'Ref_Redundant' : cma2_dict[key],
                     # 'CMA2_1_el' : d2[f'Ref - Red {combo[1]}']}        
                # d2['CMA2_1_el'] = d2[f'Ref - Red {combo[1]}']       
                # d2 ={'CMA2_1_el' : cma2_1_red_diff}        
                df2 = pd.DataFrame(data=d2)
                df2e = pd.DataFrame(data=d2e)
                df2m = pd.DataFrame(data=d2m)
                #df2.style.apply(highlight_greaterthan, axis =1)
                frame2.append(df2)
                frame2e.append(df2e)
                frame2m.append(df2m)

        # end of job loop

    # end of def
 
execute()
os.chdir(hq)
if csv:
    megaframe = pd.concat(frame)
    megaframez = pd.concat(framez)
    megaframem = pd.concat(framem)
    # megaframe.to_csv('CoordDep.csv', index=False, float_format="%.2f")
    # megaframez.to_csv('ZPVE.csv', index=False, float_format="%.2f")
    # megaframem.to_csv('e_max.csv', index=False, float_format="%.2f")
    # megaframez.to_csv('ZPVE.csv', index=False, float_format="%.8f")
    print("Final stats:")
    # print(megaframe)
    # print(np.array(megaframe.loc[:,"Ref - Nat (MP2_TZ)"]))
    for combo in combos:
        print(f'MAD ({combo[1]}):')
        print(np.mean(np.abs(np.array(megaframe.loc[:,f'Pure - Ref ({combo[1]})']))))
        print(f'mean ({combo[1]}):')
        print(np.mean(np.array(megaframe.loc[:,f'Pure - Ref ({combo[1]})'])))
        # Need the lower case max value printout here
        print(f'mean e_max Ref ({combo[1]}):')
        print(np.mean(np.array(megaframem.loc[:,f'Pure - Ref ({combo[1]})'])))
        print(f'stdev ({combo[1]}):')
        print(np.std(np.array(megaframe.loc[:,f'Pure - Ref ({combo[1]})'])))
        print(f'MAX ({combo[1]}):')
        print(np.max(np.abs(np.array(megaframe.loc[:,f'Pure - Ref ({combo[1]})']))))
        print(f'MAD ZPVE ({combo[1]}):')
        print(np.mean(np.abs(np.array(megaframez.loc[:,f'Pure - Ref ({combo[1]})']))))
        print(f'mean ZPVE ({combo[1]}):')
        print(np.mean(np.array(megaframez.loc[:,f'Pure - Ref ({combo[1]})'])))
        print(f'stdev ZPVE ({combo[1]}):')
        print(np.std(np.array(megaframez.loc[:,f'Pure - Ref ({combo[1]})'])))



        print(f'MAD CMA0 ({combo[1]}):')
        print(np.mean(np.abs(np.array(megaframe.loc[:,f'Ref - Nat ({combo[1]})']))))
        print(f'mean CMA0 ({combo[1]}):')
        print(np.mean(np.array(megaframe.loc[:,f'Ref - Nat ({combo[1]})'])))
        print(f'mean e_max CMA0 ({combo[1]}):')
        print(np.mean(np.array(megaframem.loc[:,f'Ref - Nat ({combo[1]})'])))
        print(f'stdev CMA0 ({combo[1]}):')
        print(np.std(np.array(megaframe.loc[:,f'Ref - Nat ({combo[1]})'])))
        print(f'MAX CMA0 ({combo[1]}):')
        print(np.max(np.abs(np.array(megaframe.loc[:,f'Ref - Nat ({combo[1]})']))))
        # Need MAD, mean, stdev of ZPVE here:
        print(f'MAD CMA0 ZPVE ({combo[1]}):')
        print(np.mean(np.abs(np.array(megaframez.loc[:,f'Ref - Nat ({combo[1]})']))))
        print(f'mean CMA0 ZPVE ({combo[1]}):')
        print(np.mean(np.array(megaframez.loc[:,f'Ref - Nat ({combo[1]})'])))
        print(f'stdev CMA0 ZPVE ({combo[1]}):')
        print(np.std(np.array(megaframez.loc[:,f'Ref - Nat ({combo[1]})'])))
        if n > 0:
            Eta_tab = np.array([])
            OD_tab = np.array([])
            MAD_tab = np.array([])
            RMSD_tab = np.array([])
            e_max_tab = np.array([])
            for i in range(len(xi_tol)):
                megaframe2 = pd.concat(frame2)
                megaframe2e = pd.concat(frame2e)
                megaframe2m = pd.concat(frame2m)
                CMA2dat = np.array(megaframe2.loc[:,f'Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'])
                # megaframe2.to_csv('CMA2_Convergent.csv', index=False, float_format='%.2f')
                # megaframe2e.to_csv('CMA2_e_max.csv', index=False, float_format='%.2f')
                print(f'Xi: {xi_tol[i]}')
                print('Total modes in set:')
                sum_num = np.sum(np.array(megaframe2e.loc[:,f'Natty CMA2 eta_num ({combo[1]}) xi ({xi_tol[i]})']))
                sum_denom = np.sum(np.array(megaframe2e.loc[:,f'Natty CMA2 eta_denom ({combo[1]}) xi ({xi_tol[i]})']))
                print(sum_denom)
                print(f'Eta ({combo[1]}):')
                eta = (sum_num/sum_denom)*100
                print(eta)
                Eta_tab = np.append(Eta_tab,eta)
                print('% Off diags ({combo[1]}) xi ({xi_tol[i]})')
                sum_off_diags = np.sum(np.array(megaframe2e.loc[:,f'Natty CMA2 tot_off_diags ({combo[1]}) xi ({xi_tol[i]})']))
                od = (sum_num/sum_off_diags)*100
                print(od)
                OD_tab = np.append(OD_tab,od)

                print(f'MAD CMA2 ({combo[1]}):')
                mad = np.mean(np.abs(CMA2dat))
                print(mad)
                MAD_tab = np.append(MAD_tab,mad)
                print(f'RMSD CMA2 ({combo[1]}):')
                print(np.sqrt(np.sum(CMA2dat**2)/len(CMA2dat)))
                RMSD_tab = np.append(RMSD_tab,np.sqrt(np.sum(CMA2dat**2)/len(CMA2dat)))
                print(f'Mean e_max CMA2 ({combo[1]}):')
                print(np.mean(np.array(megaframe2m.loc[:,f'Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'])))
                e_max_tab = np.append(e_max_tab,np.mean(np.array(megaframe2m.loc[:,f'Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'])))
                # print(f'mean CMA2 ({combo[1]}):')
                # print(np.mean(CMA2dat))
                # Need the lower case max value printout here
                print(f'stdev CMA2 ({combo[1]}):')
                print(np.std(CMA2dat))
                print(f'MAX CMA2 ({combo[1]}):')
                print(np.max(np.abs(CMA2dat)))
            print("\n")
            print("Figure data:")
            print("\n")
            print("xi values:")
            print(xi_tol)
            print("eta values:")
            print(Eta_tab.tolist())
            print("% off-diagonals:")
            print(OD_tab.tolist())
            print("MAD values:")
            print(MAD_tab.tolist())
            print("RMSD values:")
            print(RMSD_tab.tolist())
            print("e_max values:")
            print(e_max_tab.tolist())


# Ends SI file 
if SI:
    si.write(footer)
    si.close()

