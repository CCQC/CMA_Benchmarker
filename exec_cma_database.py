import os
import sys
import shutil
import importlib.util
import glob
import pandas as pd
import numpy as np
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
h_theory = ["CCSD_T_TZ"]
# h_theory = ["CCSD_T_QZ"]
# h_theory = ["CCSD_T_haTZ"]
# h_theory = ["CCSD_T_aTZ"]

# Theory used for off diag force constants
aux_F = ""
# aux_F = "MP2_TZ"
# aux_F = "CCSD_T_DZ"
# aux_F = "CCSD_TZ"

# CMA2 Stat Theories
# l_theory = ["B3LYP_6-31G_2df,p_"]
# l_theory = ["B3LYP_TZ"]
# l_theory = ["cas66_TZ"]
# l_theory = ["caspt2_TZ"]
# l_theory = ["dlpno_T_TZ"]
# l_theory = ["HF_TZ"]
# l_theory = ["HF_DZ"]
# l_theory = ["HF_haTZ"]
# l_theory = ["HF_aTZ"]
l_theory = ["MP2_TZ"]
# l_theory = ["MP2_QZ"]
# l_theory = ["HF_ano1"]
# l_theory = ["MP2_haDZ"]
# l_theory = ["MP2_aDZ"]
# l_theory = ["MP2_DZ"]
# l_theory = ["MP4_TZ"]
# l_theory = ["MP2_haTZ"]
# l_theory = ["MP2_aTZ"]
# l_theory = ["CCSD_T_DZ"]
# l_theory = ["CCSD_T_haDZ"]
# l_theory = ["CCSD_T_aDZ"]
# l_theory = ["CCSD_DZ"]
# l_theory = ["CCSD_TZ"]
# l_theory = ["CCSD_T_TZ"]
# l_theory = ["CCSD_T_haTZ"]
# l_theory = ["CCSD_T_aTZ"]
# l_theory = ["MP2_TZ","MP2_haTZ"]
# l_theory = ["MP2_DZ","MP2_haDZ","MP2_aDZ","MP2_TZ","MP2_haTZ","MP2_aTZ","CCSD_T_DZ","CCSD_T_haDZ","CCSD_T_aDZ","CCSD_T_TZ","CCSD_T_haTZ"]
# l_theory = ["CCSD_T_DZ","CCSD_T_haDZ"]
# l_theory = ["MP2_DZ","MP2_haDZ","MP2_aDZ","MP2_TZ","MP2_haTZ","MP2_aTZ","CCSD_T_DZ","CCSD_T_haDZ"]

combos = list(product(h_theory,l_theory))

# CMA2 Stat Theories

# cmaA_energy_regexes = [r"Grab this energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"!RHF STATE 1.\d Energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"!MCSCF STATE 1.\d Energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"!RSPT2 STATE 1.\d Energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"FINAL SINGLE POINT ENERGY\s+(\-\d+\.\d+)"]
cmaA_energy_regexes = [r"!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"!MP4\(SDTQ\)\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"E\(SCF\)\=\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"The final electronic energy is\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"Total MP2 energy\s+\=\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"!CCSD\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)",r"!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)",r"!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)",r"!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)",r"!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)",r"!MP2\s*t?o?t?a?l?\s*energy\s+(\-\d+\.\d+)",r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)",r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)",r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)",r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)",r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = [r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)",r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
# cmaA_energy_regexes = ["",""]
# cmaA_energy_regexes = ["","","","","","","","","","",""]
# cmaA_energy_regexes = ["","","","","","","",""]
# cmaA_energy_regexes = [r"\(T\)\s*t?o?t?a?l? energy\s+(\-\d+\.\d+)"]
cmaA_gradient_regex = []
cmaA_gradient_regex = [r"virial=",r"gradient"]
# cmaA_gradient_regex = [r"Total Gradient", r"tstop"]
# cmaA_success_regexes = [r"beer"]
# cmaA_success_regexes = [r"The final electronic energy is"]
cmaA_success_regexes = [r"Molpro calculation terminated"]
# cmaA_success_regexes = [r"ORCA TERMINATED NORMALLY"]
# cmaA_success_regexes = [r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated",r"Molpro calculation terminated"]
# cmaA_success_regexes = [r"Molpro calculation terminated",r"Molpro calculation terminated"]
# cmaA_success_regexes = ["",""]
# cmaA_success_regexes = ["","","","","","","","","","",""]
# cmaA_success_regexes = ["","","","","","","",""]
# cmaA_success_regexes = ["Variable memory released"]

# CMA2 Stat Theories

# Coordinates types to use
# Available: "Nattys", "Redundant", "ZMAT" (not yet tho)
# coord_type = ["Delocalized"]
coord_type = ["Nattys"]
# coord_type = ["Nattys","Nattys"]
# coord_type = ["Nattys","Nattys","Nattys","Nattys","Nattys","Nattys","Nattys","Nattys","Nattys","Nattys","Nattys"]

# Specify paths to grab data from
# Options: '/1_Closed_Shell', '/1_Linear', '/1*', '/2_Open_Shell', '/2_Linear', '/2*'
paths = ['/4*']
# job_list = ["1.2"]
# job_list = ["3.11"]
# job_list = ["4.16"]
job_list = ["4.14"]
# G3 outliers
# job_list = ["4.58","4.25","4.11","4.29","4.27","4.15","4.59","4.57","4.20","4.42","4.19","4.75","4.35"]
# job_list = ["4.35"]
# job_list = ["1.34"]
exclude_list = []
# exclude_list = ["4.16"]
# exclude_list = ["3.11","3.13","3.14","3.15","3.16"]
# exclude_list = ["3.8","3.11","3.13","3.14","3.15","3.16"]
# exclude_list = ["1.7"]


# Various output control statements
n = 0                    # Number of CMA2 corrections (n = 0 -> CMA0)
# xi_tol = [100.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.2,0.18,0.16,0.14,0.12,0.10,0.08,0.075,0.07,0.065,0.06,0.055,0.05,0.045,0.04,0.036,0.032,0.028,0.024,0.02,0.018,0.016,0.014,0.012,0.011,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0.0]    # Xi value for cutoff in determining CMA2 off diags
# xi_tol = [100,0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06,0.04,0.02,0.01,0.005]    # Xi value for cutoff in determining CMA2 off diags
# xi_tol = [0.02]
# xi_tol = [0.04]    # Xi value for cutoff in determining CMA2 off diags
xi_tol = [0.002]    # Xi value for cutoff in determining CMA2 off diags
# Azulene optimal xi tolerance
# xi_tol = [0.0017851]
xi_tol = [0.002]    # Xi value for cutoff in determining CMA2 off diags
# xi_tol = [100.0]
# xi_tol = []    # Xi value for cutoff in determining CMA2 off diags
# xi_tol = [0.07]    # Xi value for cutoff in determining CMA2 off diags
# xi_tol = [0.00275]    # Xi value for cutoff in determining CMA2 off diags
omega_tol = []    # Omega value for cutoff in determining CMA3 off diags
# Azulene optimal omega tolerance
# omega_tol = [0.0676105]
# omega_tol = [0.0676109701537725]
omega_tol = [0.07]
# omega_tol = [0.4]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [0.065,1.0,2.5,10.0,200.0,300.0]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [0.07,0.0675,0.065]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [0.073]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [0.065]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [100,75,50,25]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [0.1]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [1000,300.0,200.0,100.0,20.0,10.0,5.0,1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]    # Omega value for cutoff in determining CMA3 off diags
# omega_tol = [1000,300.0,200.0,100.0,20.0,10.0,5.0,2.0,1.0,0.5,0.25,0.1,0.05,0.02,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]    # Omega value for cutoff in determining CMA3 off diags
# od_inds = [[1,2],[0,3],[3,5],[21,28]]
# od_inds = [[0,2],[10,16],[21,28]]
#od_inds = [[2,4],[1,3],[22,26],[0,3]]
od_inds = []
# Azulene 38 necessary off-diagonals for CMA-1
# od_inds = [[6, 25], [14, 25], [25, 26], [6, 28], [14, 28], [25, 28], [26, 28], [25, 31], [28, 31], [6, 33], [14, 33], [28, 33], [25,33], [31, 33], [6, 35], [14, 35], [25, 35], [26, 35], [28, 35], [31, 35], [33, 35], [6, 37], [14, 37], [25, 37], [26, 37], [28, 37], [33, 37], [35, 37], [6, 39], [14, 39], [25, 39], [26, 39], [28, 39], [30, 39], [31, 39], [33, 39], [35, 39], [37, 39]]
# od_inds = [[14,39],[14,25],[25,28],[25,39],[25,26],[26,28],[28,30],[33,35],[37,39]]         # Contains a list of lists, where the sublists contain off-diagonal elements to be computed in CMA-1
# od_inds = [[0,3],[0,5],[3,5]]
# od_inds = [[0,3],[1,4],[21,28]]
# od_inds = [[0,3],[1,4],[4,5],[21,28]]
#cmaA = False             # Run CMA_B instead of CMA_A
cmaA = True             # Run CMA_A instead of CMA_B
# csv = False               # Generate database .csv file
csv = True               # Generate database .csv file
SI = False                # Generate LaTeX SI file
# SI = True               # Generate LaTeX SI file
compute_all = False       # run calculations for all or a select few
# compute_all = True       # run calculations for all or a select few
off_diag = 0   # Set this option for CMA0
# off_diag = 1   # Set this option for CMA1. Additional off-diagonal elements will need to be specified using ___.
# off_diag = 2   # Set this option for CMA2. Off-diags will be auto generated, but an aux hessian will need be specified using ___.
# off_diag = 3   # Set this option for CMA3. Off-diags will be auto generated, but an aux hessian will need be specified using ___.
deriv_level = 0         # (CMA_A) if 0, compute initial hessian by singlepoints. If 1, compute initial hessian with findif of gradients
# deriv_level = 1         # (CMA_A) if 0, compute initial hessian by singlepoints. If 1, compute initial hessian with findif of gradients
# second_order = True    # If True, read in cartesian gradient and force constant info to be converted to internal coordinates.
second_order = False    # If False, generate displacements to manually compute the CMA-0A internal coord force constants.
# coord_type_b = "cartesian" # Toggle this for type of coordinate used in inital force constant computations
coord_type_b = "internal" # Toggle this for type of coordinate used in inital force constant computations

# =====================
# Some useful functions
# =====================

def freq_diff(reference, calculated):
    """Return the frequency difference, calculated minus reference."""
    return calculated - reference


def return_path_base(jobpath):
    """Return the final directory name from a job path."""
    return os.path.basename(os.path.normpath(jobpath))


def collect_job_paths(root, patterns, selected_jobs, exclude_jobs, compute_all):
    """Build the ordered list of jobs to process."""
    if compute_all:
        jobs = []
        path_indices = []
        for pattern in patterns:
            path_indices.append(len(jobs))
            matches = glob.glob(os.path.join(root, pattern, '[1-9]*_*/'))
            matches.sort(key=lambda p: int(re.search(r'/\d_.*/(\d*)_.*', p).group(1)))
            jobs.extend(matches)

        excluded = []
        for job in exclude_jobs:
            id1, id2 = job.split('.')
            excluded.extend(glob.glob(os.path.join(root, f'{id1}_*', f'{id2}_*/')))
        jobs = [job for job in jobs if job not in excluded]
        return jobs, path_indices

    jobs = []
    for job in selected_jobs:
        id1, id2 = job.split('.')
        jobs.extend(glob.glob(os.path.join(root, f'{id1}_*', f'{id2}_*/')))
    return jobs, []


def print_job_list(jobs, patterns, path_indices):
    """Print the jobs in the same compact format used by the original script."""
    print(f"Generating database entries for {len(jobs)} jobs:", end="")
    count = 0
    for i, job in enumerate(jobs):
        if i in path_indices:
            print(f"\n\nDirectory: {patterns[path_indices.index(i)]}", end="")
            count = 0
        prefix = "\n" if count % 5 == 0 else ""
        print(f"{prefix}{return_path_base(job):20}", end="")
        count += 1
    print("\n")


def initialize_data_frames():
    """Create the lists used to accumulate database DataFrames."""
    frames = {
        'main': [],
        'zpve': [],
        'max': [],
        'corr': [],
        'corr_eta': [],
        'corr_max': [],
    }
    return frames


def append_dataframe_triplet(targets, data, zdata, mdata):
    """Append the three standard result dictionaries as DataFrames."""
    targets[0].append(pd.DataFrame(data=data))
    targets[1].append(pd.DataFrame(data=zdata))
    targets[2].append(pd.DataFrame(data=mdata))


def safe_remove(*filenames):
    """Remove files if present without changing the normal workflow."""
    for filename in filenames:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass


def rmsd(values):
    """Return the root-mean-square value of an array."""
    values = np.asarray(values)
    return np.sqrt(np.mean(values ** 2))


def print_basic_statistics(megaframe, megaframez, megaframem, combo):
    """Print the standard CMA/reference statistics for one theory combination."""
    theory = combo[1]
    natty_denom = np.asarray(megaframe[f'Natty denom ({theory})'])
    pure_ref = np.asarray(megaframe[f'Pure - Ref ({theory})'])
    pure_ref_z = np.asarray(megaframez[f'Pure - Ref ({theory})'])
    pure_ref_emax = np.asarray(megaframem[f'Pure - Ref ({theory})'])
    ref_nat = np.asarray(megaframe[f'Ref - Nat ({theory})'])
    ref_nat_z = np.asarray(megaframez[f'Ref - Nat ({theory})'])
    ref_nat_emax = np.asarray(megaframem[f'Ref - Nat ({theory})'])

    print('Total modes in set:')
    print(len(natty_denom))
    print(f'MAD ({theory}):')
    print(np.mean(np.abs(pure_ref)))
    print(f'mean ({theory}):')
    print(np.mean(pure_ref))
    print(f'mean e_max Ref ({theory}):')
    print(np.mean(pure_ref_emax))
    print(f'stdev ({theory}):')
    print(np.std(pure_ref))
    print(f'MAX ({theory}):')
    print(np.max(np.abs(pure_ref)))
    print('Percent pure outliers:')
    print(np.sum(np.asarray(megaframez[f'Pure outliers ({theory})'])) * 100.0 / len(natty_denom))
    print(f'MAD ZPVE ({theory}):')
    print(np.mean(np.abs(pure_ref_z)))
    print(f'mean ZPVE ({theory}):')
    print(np.mean(pure_ref_z))
    print(f'stdev ZPVE ({theory}):')
    print(np.std(pure_ref_z))
    print("")

    print(f'MAD CMA0 ({theory}):')
    print(np.mean(np.abs(ref_nat)))
    print(f'mean CMA0 ({theory}):')
    print(np.mean(ref_nat))
    print(f'mean e_max CMA0 ({theory}):')
    print(np.mean(ref_nat_emax))
    print(f'stdev CMA0 ({theory}):')
    print(np.std(ref_nat))
    print(f'MAX CMA0 ({theory}):')
    print(np.max(np.abs(ref_nat)))
    print('Percent CMA-0 outliers:')
    print(np.sum(np.asarray(megaframez[f'Natty outliers ({theory})'])) * 100.0 / len(natty_denom))
    print(f'MAD CMA0 ZPVE ({theory}):')
    print(np.mean(np.abs(ref_nat_z)))
    print(f'mean CMA0 ZPVE ({theory}):')
    print(np.mean(ref_nat_z))
    print(f'stdev CMA0 ZPVE ({theory}):')
    print(np.std(ref_nat_z))


def print_off_diagonal_statistics(frame2, frame2e, frame2m, combo, kind, tolerances):
    """Print CMA2/CMA3 statistics and return data used for figures."""
    theory = combo[1]
    label = 'xi' if kind == 2 else 'omega'
    eta_values, od_values, mad_values, rmsd_values, emax_values = [], [], [], [], []
    std_values, max_values = [], []

    megaframe2 = pd.concat(frame2)
    megaframe2e = pd.concat(frame2e)
    megaframe2m = pd.concat(frame2m)

    for tolerance in tolerances:
        frequency_column = f'Ref - Natty CMA{kind} ({theory}) {label} ({tolerance})'
        data = np.asarray(megaframe2[frequency_column])
        eta_num = np.sum(np.asarray(megaframe2e[f'Natty CMA{kind} eta_num ({theory}) {label} ({tolerance})']))
        eta_denom = np.sum(np.asarray(megaframe2e[f'Natty CMA{kind} eta_denom ({theory}) {label} ({tolerance})']))
        total_off_diags = np.sum(np.asarray(megaframe2e[f'Natty CMA{kind} tot_off_diags ({theory}) {label} ({tolerance})']))

        print(f'{"Xi" if kind == 2 else "Omega"}: {tolerance}')
        print('Total modes in set:')
        print(eta_denom)
        print(f'Eta ({theory}):')
        eta = eta_num / eta_denom * 100
        print(eta)
        eta_values.append(eta)
        print(f'% Off diags ({theory}) {label} ({tolerance})')
        od = eta_num / total_off_diags * 100
        print(od)
        od_values.append(od)
        print(f'MAD CMA{kind} ({theory}):')
        mad = np.mean(np.abs(data))
        print(mad)
        mad_values.append(mad)
        print(f'RMSD CMA{kind} ({theory}):')
        current_rmsd = rmsd(data)
        print(current_rmsd)
        rmsd_values.append(current_rmsd)
        print(f'Mean e_max CMA{kind} ({theory}):')
        emax = np.mean(np.asarray(megaframe2m[f'Ref - Natty CMA{kind} ({theory}) {label} ({tolerance})']))
        print(emax)
        emax_values.append(emax)
        print(f'stdev CMA{kind} ({theory}):')
        std = np.std(data)
        print(std)
        print(f'MAX CMA{kind} ({theory}):')
        maximum = np.max(np.abs(data))
        print(maximum)
        std_values.append(std)
        max_values.append(maximum)

    print("\n")
    print("Figure data:")
    print("\n")
    print(f'{label} values:')
    print(tolerances)
    print("eta values:")
    print(eta_values)
    print("% off-diagonals:")
    print(od_values)
    print("MAD values:")
    print(mad_values)
    print("RMSD values:")
    print(rmsd_values)
    print("e_max values:")
    print(emax_values)
    if kind == 3:
        print("stdev values:")
        print(std_values)
        print("MAX values:")
        print(max_values)

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



print("Contributors: Dr. Mitchell Lahm, Dr. Nathaniel Kitzmiller, Dr. Henry Mull, Dr. Laura N. O. Dornshuld, Jace Jin")
print()
print("Combinations of levels of theory (high, low): ", end="")
print(*combos, sep=", ")
print("Coordinate types: ", end="")
print(*coord_type, sep=", ")
print(f"Number of off-diagonal corrections: {n}")
print()

# Grab jobs from each path and order them numerically by ID
jobb_list, path_ind = collect_job_paths(
    hq, paths, job_list, exclude_list, compute_all
)
print_job_list(jobb_list, paths, path_ind)

# Initialize frames for pandas database
frame, framez, framem = [], [], []
if off_diag > 0:
    frame2, frame2e, frame2m = [], [], []

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
            d2 = {'Molecule' : None}
            d2e = {'Molecule' : None}
            d2m = {'Molecule' : None}

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
                    elif coord == "Delocalized":
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
                            shutil.copyfile(job + combo[0] + "/zmat_cmaA", job + "zmat2")
                            execMerger.options.coords = 'Custom'
                        else:
                            shutil.copyfile(job + combo[0] + "/zmat", job + "zmat2")
                            
                
                    execMerger.options.n_cma2 = n
                    execMerger.options.off_diag = off_diag
                
                    # Run CMA
                    execMerger.run(execMerger.options, Proj, sym_sort=sym_sort, coord_type_b=coord_type_b)
                
                    # Collect data
                    if coord_type.index(coord) == 0:
                        d[f'Ref ({combo[0]})'] = execMerger.reference_freq
                        z[f'Ref ({combo[0]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133)
                        mol.freqs[f'Ref ({combo[0]})'] = execMerger.reference_freq
                        d[f'Ref ({combo[1]})'] = execMerger.ref_b
                        z[f'Ref ({combo[1]})'] = np.sum(execMerger.ref_b)/(2*349.7550881133)
                        mol.freqs[f'Ref ({combo[1]})'] = execMerger.ref_b
                        

                        # Number the modes
                        d['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                        z['Molecule'] = [f"{mol.name} ({mol.ID})"]
                
                    if coord == "Nattys":
                        d[f'Natty ({combo[1]})'] = execMerger.Freq_CMA0
                        z[f'Natty ({combo[1]})'] = np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                        d[f'Ref - Nat ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_CMA0)
                        z[f'Ref - Nat ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                        mol.freqs[f'Natty ({combo[1]})'] = execMerger.Freq_CMA0
                    if coord == "Delocalized":
                        if 'Linear' not in job:
                            d[f'Red ({combo[1]})'] = execMerger.Freq_delocalized
                            z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_delocalized)/(2*349.7550881133)
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_delocalized)
                            z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_delocalized)/(2*349.7550881133)
                            mol.freqs[f'Red ({combo[1]})'] = execMerger.Freq_delocalized
                        else:
                            d[f'Red ({combo[1]})'] = execMerger.Freq_CMA0
                            z[f'Red ({combo[1]})'] = np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_CMA0)
                            z[f'Ref - Red ({combo[1]})'] = np.sum(execMerger.reference_freq)/(2*349.7550881133) - np.sum(execMerger.Freq_CMA0)/(2*349.7550881133)
                            mol.freqs[f'Red ({combo[1]})'] = execMerger.Freq_CMA0
                
                    # Collect data for CMA2
                    if off_diag > 0:
                        if coord == "Nattys":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                            d2[f"Ref {combo[0]}"] = execMerger.reference_freq
                            d2[f'Natty ({combo[1]})'] = execMerger.Freq_CMA0
                            cma2_freqs_natty = execMerger.Freq_cma2 
                          
                            d2[f'Natty CMA2 ({combo[1]})'] = cma2_freqs_natty 
                            d2[f'Ref - Natty ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_CMA0)
                            d2[f'Ref - Natty CMA2 ({combo[1]})'] = freq_diff(execMerger.reference_freq, cma2_freqs_natty)
                            print('Give us CMA2!')                   
                        if coord == "Delocalized":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                            d2[f"Ref {combo[0]}"] = execMerger.reference_freq
                            if 'Linear' not in job:
                                d2[f'Red ({combo[1]})'] = execMerger.Freq_delocalized
                            else:
                                d2[f'Red ({combo[1]})'] = execMerger.Freq_CMA0
                            cma2_freqs_red = execMerger.Freq_cma2 
                          
                            d2[f'Natty CMA2 ({combo[1]})'] = cma2_freqs_red 
                            if 'Linear' not in job:
                                d2[f'Ref - Red ({combo[1]})'] = freq_diff(execMerger.reference_freq, execMerger.Freq_delocalized)
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

                # Difference between Natty and Delocalized freqs
                if 'Nattys' in coord_type and 'Delocalized' in coord_type:
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
                    execMerger = Merger(cmaA_path= "/" + combo[0]+"/Disps_" + combo[1])
                    if os.path.exists(os.getcwd() + "/" + combo[0]+"/Disps_" + combo[1] + "/templateInit.dat"):
                        #change to True if you need the displacements generated
                        execMerger.options.calc_b = True
                        # execMerger.options.calc_b = False

                    if os.path.exists(os.getcwd() + "/" + combo[0]+"/Disps_" + combo[1] + "/DispsB"):
                        execMerger.options.calc_b = False
                        execMerger.options.gen_disps_b = False
                    
                    execMerger.options.cart_insert_b = 9
                    # execMerger.options.cart_insert_b = 10
                    execMerger.options.other_F_matrix_del = ''
                    if combo[1] == "B3LYP_6-31G_2df,p_":
                        execMerger.options.other_F_matrix = 'HF_6-31G_2df,p_'
                    elif combo[1] == "CCSD_T_DZ":
                        execMerger.options.other_F_matrix = 'HF_DZ'
                        # execMerger.options.other_F_matrix_del = 'HF_TZ'
                        # execMerger.options.other_F_matrix = 'MP2_DZ'
                        # execMerger.options.other_F_matrix_del = 'MP2_TZ'
                        # execMerger.options.other_F_matrix = 'B3LYP_6-31G_2df,p_'
                    elif combo[1] == "CCSD_T_haDZ":
                        execMerger.options.other_F_matrix = 'CCSD_T_DZ'
                    elif combo[1] == "MP2_TZ":
                        # execMerger.options.other_F_matrix = 'CCSD_T_TZ'
                        # execMerger.options.other_F_matrix = 'B3LYP_6-31G_2df,p_'
                        execMerger.options.other_F_matrix = 'HF_TZ'
                        # execMerger.options.other_F_matrix_del = 'CCSD_T_DZ'
                    elif combo[1] == "MP2_haTZ":
                        execMerger.options.other_F_matrix = 'HF_haTZ'
                    elif combo[1] == "MP2_aTZ":
                        execMerger.options.other_F_matrix = 'HF_aTZ'
                    elif combo[1] == "MP2_ano1":
                        execMerger.options.other_F_matrix = 'HF_ano1'
                    else:
                        execMerger.options.other_F_matrix = ''

                    execMerger.options.aux_F = aux_F
                    # execMerger.options.other_F_matrix = 'MP2_TZ'
                    if len(execMerger.options.other_F_matrix):
                        if os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_int_b.dat") and coord_type_b == 'internal':
                            shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_int_b.dat",os.getcwd()+"/inter_fc.dat")
                        elif os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_cart_b.dat") and coord_type_b == 'cartesian':
                            shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_cart_b.dat",os.getcwd()+"/inter_fc_cart.dat")
                            if os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_cart_b.grad"):
                                shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix+"/fc_cart_b.grad",os.getcwd()+"/inter_fc_cart.grad")
                            else:
                                print("A gradient is necessary for this transformation.")
                                raise RuntimeError
                    
                    if len(execMerger.options.other_F_matrix_del):
                        if os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix_del+"/fc_int_b.dat") and coord_type_b == 'internal':
                            shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix_del+"/fc_int_b.dat",os.getcwd()+"/inter_fc_del.dat")
                        elif os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix_del+"/fc_cart_b.dat") and coord_type_b == 'cartesian':
                            shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix_del+"/fc_cart_b.dat",os.getcwd()+"/inter_fc_cart_del.dat")
                            if os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix_del+"/fc_cart_b.grad"):
                                shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.other_F_matrix_del+"/fc_cart_b.grad",os.getcwd()+"/inter_fc_cart_del.grad")
                            else:
                                print("A gradient is necessary for this transformation.")
                                raise RuntimeError
                    
                    
                    
                    if len(execMerger.options.aux_F):
                        if os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.aux_F+"/fc_int_nat.dat") and coord_type_b == 'internal':
                            shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.aux_F+"/fc_int_nat.dat",os.getcwd()+"/aux_fc.dat")
                        elif os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.aux_F+"/fc_cart.dat") and coord_type_b == 'cartesian':
                            shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.aux_F+"/fc_cart.dat",os.getcwd()+"/aux_fc_cart.dat")
                            if os.path.exists(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.aux_F+"/fc_cart.grad"):
                                shutil.copyfile(os.getcwd()+"/"+combo[0]+"/Disps_"+execMerger.options.aux_F+"/fc_cart.grad",os.getcwd()+"/aux_fc_cart.grad")
                            else:
                                print("A gradient is necessary for this transformation.")
                                raise RuntimeError

                    if combo[1] == "CCSD_T_DZ":
                        execMerger.options.cart_insert_b = 9
                    elif combo[1] == "B3LYP_6-31G_2df,p_" or combo[1] == "HF_6-31G_2df,p_" or combo[1] == "df_MP2_TZ" or combo[1] == "B3LYP_TZ" or combo[1] == "dlpno_T_TZ":
                        execMerger.options.cart_insert_b = 4
                        execMerger.options.program_b = "psi4@master"
                    execMerger.options.coords = coord
                    execMerger.options.n_cma2 = n
                    execMerger.options.off_diag = off_diag
                    execMerger.options.deriv_level_b = deriv_level
                    execMerger.options.second_order = second_order
                    execMerger.options.scaled_disp = True
                    # execMerger.options.program_b = "cfour@str"
                    # execMerger.options.program_b = "molpro"
                    sym_sort = np.array([])
                    tiles = np.array([])
                    tile_type = np.array([])
                    tile_xi = {}
                    if coord == "Nattys":
                        if coord_type_b == 'cartesian':
                            try:
                                shutil.copyfile(job + combo[0] + "/Disps_" + combo[1] + "/fc_cart.dat", job + "fc.dat")
                                shutil.copyfile(job + combo[0] + "/Disps_" + combo[1] + "/fc_cart.grad", job + "fc.grad")
                            except:
                                print('Once again, the directory does not contain the sufficient files for the specified job')
                        try: 
                            shutil.copyfile(job + combo[0] + "/zmat", job + "zmat")
                            shutil.copyfile(job + combo[0] + "/zmat", job + "zmat2")
                            shutil.copyfile(job + combo[0] + "/fc.dat", job + "fc2.dat")      
                        except:
                            print('Once again, the directory does not contain the sufficient files for the specified job')
                            mol.direc_complete = False
                            break 
                        cmaA_coord = "nat"
                        execMerger.options.man_proj = True
                        execMerger.options.coords = 'Custom'
                        execMerger.options.gradient_regex_b = cmaA_gradient_regex
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
                        try:
                            tiles = copy.copy(project_obj.tiles)
                            tile_type = copy.copy(project_obj.tile_type)
                            tile_xi = copy.copy(project_obj.tile_xi)
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
                            #shutil.copyfile(job + combo[0] + "/zmat_cmaA", job + "zmat")
                            #shutil.copyfile(job + combo[0] + "/zmat_cmaA_Final", job + "zmat2")
                        except:
                            print('Once again, the directory does not contain the sufficient files for the specified job')
                            mol.direc_complete = False
                            break 
                        cmaA_coord = "red"
                        execMerger.options.man_proj = False
                        execMerger.options.coords = coord
                        execMerger.options.gradient_regex = cmaA_gradient_regex
                        Proj = None
                        if 'Linear' in job:
                            execMerger.options.coords = 'Custom'
                    execMerger.run(execMerger.options,Proj,energy_regex=cmaA_energy_regexes[countt],success_regex=cmaA_success_regexes[countt],cmaA_coord=cmaA_coord, sym_sort=sym_sort, xi_tol=xi_tol, omega_tol=omega_tol, coord_type_b=coord_type_b, od_inds=od_inds, tiles=tiles, tile_type=tile_type, tile_xi=tile_xi)
                    
                    ref_freq = execMerger.reference_freq.copy()
                    freq_indices = [i for i in range(len(ref_freq))]
                    freq_indices = np.array(freq_indices)
                    
                    # Collect data
                    if coord_type.index(coord) == 0:
                        ref_freq = execMerger.reference_freq.copy()
                        ref_freq_b = execMerger.ref_b.copy()
                        
                        d[f'Ref ({combo[0]})'] = ref_freq
                        z[f'Ref ({combo[0]})'] = np.sum(execMerger.reference_freq)/2
                        mol.freqs[f'Ref ({combo[0]})'] = ref_freq
                        mol.resid[f'Ref ({combo[0]})'] = ref_freq
                        
                        d[f'Ref ({combo[1]})'] = ref_freq_b
                        d[f'Pure - Ref ({combo[1]})'] = freq_diff(ref_freq, ref_freq_b)
                        d[f'ABS Pure - Ref ({combo[1]})'] = np.abs(freq_diff(ref_freq, ref_freq_b))
                        z[f'Ref ({combo[1]})'] = np.sum(ref_freq_b)/2
                        z[f'Pure outliers ({combo[1]})'] = execMerger.pure_outliers
                        z[f'Pure - Ref ({combo[1]})'] = np.sum(ref_freq_b)/2 - np.sum(ref_freq)/2
                        m[f'Pure - Ref ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, ref_freq_b)))
                        mol.freqs[f'Initial ({combo[1]})'] = ref_freq_b

                        # Number the modes
                        d['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(execMerger.reference_freq))]
                        z['Molecule'] = [f"{mol.name} ({mol.ID})"]
                    
                    if coord == "Nattys":
                        d['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in freq_indices]
                        z['Molecule'] = [f"{mol.name} ({mol.ID})"]
                        m['Molecule'] = [f"{mol.name} ({mol.ID})"]
                        custom_freq = execMerger.Freq_CMA0.copy()
                        d[f'Natty denom ({combo[1]})'] = execMerger.denom
                        d[f'Natty ({combo[1]})'] = custom_freq
                        z[f'Natty ({combo[1]})'] = np.sum(custom_freq)/2
                        z[f'Natty outliers ({combo[1]})'] = execMerger.outliers
                        d[f'Ref - Nat ({combo[1]})'] = freq_diff(ref_freq, custom_freq)
                        d[f'ABS Ref - Nat ({combo[1]})'] = np.abs(freq_diff(ref_freq, custom_freq))
                        z[f'Ref - Nat ({combo[1]})'] = np.sum(custom_freq)/2 - np.sum(ref_freq)/2
                        m[f'Ref - Nat ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, custom_freq)))
                        mol.freqs[f'Natty ({combo[1]})'] = custom_freq
                        # Turn this back on after assembling SI
                        mol.resid[f'Natty ({combo[1]})'] = freq_diff(ref_freq, custom_freq)
                    if coord == "Delocalized":
                        if 'Linear' not in job:
                            red_freq = execMerger.Freq_delocalized.copy()

                            d[f'Red ({combo[1]})'] = red_freq
                            z[f'Red ({combo[1]})'] = np.sum(red_freq)/2
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, red_freq)
                            d[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, red_freq))
                            z[f'Ref - Red ({combo[1]})'] = np.sum(ref_freq)/2 - np.sum(red_freq)/2
                            m[f'Ref - Red ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, red_freq)))
                            mol.freqs[f'Red ({combo[1]})'] = red_freq
                            mol.resid[f'Red ({combo[1]})'] = freq_diff(ref_freq, red_freq)
                        else:
                            cust_freq = execMerger.Freq_CMA0.copy()
                            d[f'Red ({combo[1]})'] = cust_freq
                            z[f'Red ({combo[1]})'] = np.sum(cust_freq)/2
                            d[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, cust_freq)
                            d[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, cust_freq))
                            z[f'Ref - Red ({combo[1]})'] = np.sum(ref_freq)/2 - np.sum(cust_freq)/2
                            m[f'Ref - Red ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, cust_freq)))
                            mol.freqs[f'Red ({combo[1]})'] = cust_freq
                            mol.resid[f'Red ({combo[1]})'] = freq_diff(ref_freq, cust_freq)
                    
                    #Collect data for CMA2
                    if off_diag > 0:

                        if coord == "Nattys":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(ref_freq))]
                            d2e['Molecule'] = [f"{mol.name} ({mol.ID})"]
                            d2m['Molecule'] = [f"{mol.name} ({mol.ID})"]
                            d2[f"Ref {combo[0]}"] = ref_freq
                            d2[f'Natty ({combo[1]})'] = custom_freq
                            if off_diag == 1:
                                cmaA_freqs_natty = execMerger.Freq_cmaA.copy()
                                mol.freqs[f'Natty CMA1 ({combo[1]})'] = cmaA_freqs_natty
                                mol.resid[f'Natty CMA1 ({combo[1]})'] = freq_diff(ref_freq, cmaA_freqs_natty)
                                d2[f'Natty CMA1 ({combo[1]})'] = cmaA_freqs_natty
                                d2[f'Ref - Natty CMA1 ({combo[1]})'] = freq_diff(ref_freq, cmaA_freqs_natty)
                                d2[f'ABS Ref - Natty CMA1 ({combo[1]})'] = np.abs(freq_diff(ref_freq, cmaA_freqs_natty))
                                d2m[f'Ref - Natty CMA1 ({combo[1]})'] = np.max(np.abs(freq_diff(ref_freq, cmaA_freqs_natty)))
                            elif off_diag == 2:
                                cma2_freqs_natty = execMerger.Freq_cma2.copy()
                                for i in range(len(xi_tol)):
                                    mol.freqs[f'Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = cma2_freqs_natty[i]
                                    mol.resid[f'Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = freq_diff(ref_freq, cma2_freqs_natty[i])
                                    d2[f'Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = cma2_freqs_natty[i] 
                                    d2[f'Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = freq_diff(ref_freq, cma2_freqs_natty[i])
                                    d2[f'ABS Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = np.abs(freq_diff(ref_freq, cma2_freqs_natty[i]))
                                    d2e[f'Natty CMA2 eta_num ({combo[1]}) xi ({xi_tol[i]})'] = execMerger.eta_num[i]
                                    d2e[f'Natty CMA2 eta_denom ({combo[1]}) xi ({xi_tol[i]})'] = execMerger.eta_denom[i]
                                    d2e[f'Natty CMA2 tot_off_diags ({combo[1]}) xi ({xi_tol[i]})'] = execMerger.total_off_diags[i]
                                    d2m[f'Ref - Natty CMA2 ({combo[1]}) xi ({xi_tol[i]})'] = np.max(np.abs(freq_diff(ref_freq, cma2_freqs_natty[i])))
                            elif off_diag == 3:
                                cma3_freqs_natty = execMerger.Freq_cma3.copy()
                                for i in range(len(omega_tol)):
                                    mol.freqs[f'Natty CMA3 ({combo[1]}) omega ({omega_tol[i]})'] = cma3_freqs_natty[i]
                                    mol.resid[f'Natty CMA3 ({combo[1]}) omega ({omega_tol[i]})'] = freq_diff(ref_freq, cma3_freqs_natty[i])
                                    d2[f'Natty CMA3 ({combo[1]}) omega ({omega_tol[i]})'] = cma3_freqs_natty[i] 
                                    d2[f'Ref - Natty CMA3 ({combo[1]}) omega ({omega_tol[i]})'] = freq_diff(ref_freq, cma3_freqs_natty[i])
                                    d2[f'ABS Ref - Natty CMA3 ({combo[1]}) omega ({omega_tol[i]})'] = np.abs(freq_diff(ref_freq, cma3_freqs_natty[i]))
                                    d2e[f'Natty CMA3 eta_num ({combo[1]}) omega ({omega_tol[i]})'] = execMerger.eta_num[i]
                                    d2e[f'Natty CMA3 eta_denom ({combo[1]}) omega ({omega_tol[i]})'] = execMerger.eta_denom[i]
                                    d2e[f'Natty CMA3 tot_off_diags ({combo[1]}) omega ({omega_tol[i]})'] = execMerger.total_off_diags[i]
                                    d2m[f'Ref - Natty CMA3 ({combo[1]}) omega ({omega_tol[i]})'] = np.max(np.abs(freq_diff(ref_freq, cma3_freqs_natty[i])))
                        if coord == "Delocalized":
                            d2['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(ref_freq))]
                            d2e['Molecule'] = [f"{mol.name} ({mol.ID}) mode {i+1}" for i in range(len(ref_freq))]
                            d2[f"Ref {combo[0]}"] = ref_freq
                            if 'Linear' not in job:
                                d2[f'Red ({combo[1]})'] = red_freq
                                d2[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, red_freq)
                                d2[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, red_freq))
                            else:
                                d2[f'Red ({combo[1]})'] = cust_freq
                                d2[f'Ref - Red ({combo[1]})'] = freq_diff(ref_freq, cust_freq)
                                d2[f'ABS Ref - Red ({combo[1]})'] = np.abs(freq_diff(ref_freq, cust_freq))
                            cma2_freqs_red = execMerger.Freq_cma2 
                          
                            d2[f'Natty CMA2 ({combo[1]})'] = cma2_freqs_red 
                            d2[f'Ref - Red CMA2 ({combo[1]})'] = freq_diff(ref_freq, cma2_freqs_red)
                            d2[f'ABS Ref - Red CMA2 ({combo[1]})'] = np.abs(freq_diff(ref_freq, cma2_freqs_red))


                    del execMerger
                    del Merger
            if 'Nattys' in coord_type and 'Delocalized' in coord_type:
                d[f'Nat - Red {combo[1]}'] = freq_diff(d[f'Natty ({combo[1]})'], d[f'Red ({combo[1]})'])
            countt += 1

        # end of combo loop
        if mol.direc_complete: 
            # Print molecule information
            print("begin:")
            if SI:
                si.write(mol.build_latex_output(cmaA=cmaA,combos=combos,xi_tol=xi_tol))
            
            # Clean up job directory
            if not cmaA:
                os.remove("fc.dat")
            try: 
                os.remove("zmat")
                os.remove("zmat2")
                if second_order:
                    os.remove("fc.dat")
                    os.remove("fc.grad")
                    if off_diag == 2 or off_diag == 3:
                        os.remove("inter_fc_cart.dat")
                        os.remove("inter_fc_cart.grad")
                    if len(aux_F):
                        os.remove("aux_fc_cart.dat")
                        os.remove("aux_fc_cart.grad")
                os.remove("fc2.dat")
                if coord_type_b == 'internal':
                    os.remove("inter_fc.dat")
                    if len(aux_F):
                        os.remove("aux_fc.dat")
            except:
                print('These are not the files you are looking for') 
            
            if coord_type[0] == 'Nattys':
                print("end:")
                mol.run()
            if coord_type[0] == 'Delocalized':
                print("We're trying it out!")
                mol.run()
            sys.path.remove(job)
            del mol


            # Add to pandas dataframe
            if csv:
                append_dataframe_triplet((frame, framez, framem), d, z, m)

            if off_diag > 0:
                append_dataframe_triplet((frame2, frame2e, frame2m), d2, d2e, d2m)

        # end of job loop

    # end of def
 
execute()
os.chdir(hq)

if csv:
    megaframe = pd.concat(frame)
    megaframez = pd.concat(framez)
    megaframem = pd.concat(framem)

    print("Final stats:")
    for combo in combos:
        print_basic_statistics(megaframe, megaframez, megaframem, combo)
        print("off_diag:")
        print(off_diag)

        if off_diag == 2:
            print_off_diagonal_statistics(
                frame2, frame2e, frame2m, combo, 2, xi_tol
            )
        elif off_diag == 3:
            print_off_diagonal_statistics(
                frame2, frame2e, frame2m, combo, 3, omega_tol
            )


# Ends SI file 
if SI:
    si.write(footer)
    si.close()


