import os
import re
import sys
import shutil
import json
import subprocess
import time
import numpy as np
import copy

from numpy import linalg as LA
from numpy.linalg import inv
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag
from concordantmodes.algorithm import Algorithm
from concordantmodes.directory_tree import DirectoryTree
from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.force_constant import ForceConstant
from concordantmodes.gf_method import GFMethod
from concordantmodes.g_matrix import GMatrix
from concordantmodes.g_read import GrRead
from concordantmodes.int2cart import Int2Cart
from concordantmodes.options import Options
from concordantmodes.reap import Reap
from concordantmodes.s_vectors import SVectors
from concordantmodes.submit import Submit
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.zmat import Zmat

import copy
from fractions import Fraction

class Merger(object):

    def __init__(self, cma1_path=None):
        options_kwargs = {
            "queue": "gen4.q,gen6.q,gen5.q",
            "program": "molpro@2010.1.67+mpi",
            "energy_regex": r"\(T\) energy\s+(\-\d+\.\d+)",
            'energy_regex' : r"\s*\!CCSD\(T\) total energy\s+(-\d+\.\d+)",
            "cart_insert": 9,
            "calc" : False,
            "calc_init" : False,
            "success_regex": r"Variable memory released",
            # "disp_points" : "5",
            # "reduced_disp" : True,
            # "cart_insert" : 26,
            # "disp" : 1.0
            # "disp" : 0.005
            # "disp" : [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
            # "disp" : [0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.04,0.04]
            # "disp" : [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.04,0.04,0.02]
            # "disp" : [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.04,0.04,0.04,0.04,0.04]
        }
        options_obj = Options(**options_kwargs)
        self.options = options_obj 
        self.cma1_path = cma1_path
    # function that returns diagonal fc matrix + n-largest off-diagonal elements
    def run(self, opts, Proj, energy_regex=None, success_regex=None, cma1_coord=None, sym_sort=None, xi_tol=[], coord_type_init="internal", od_inds=[]):
        
        self.coord_type_init = coord_type_init
        
        self.Proj = Proj
        
        if len(sym_sort) > 1:
            flat_sym_sort = np.array([])
            for i in range(len(sym_sort)):
                flat_sym_sort = np.append(flat_sym_sort,sym_sort[i])
            flat_sym_sort = flat_sym_sort.astype(int)
        
        self.options = opts 
        rootdir = os.getcwd()
        zmat_obj = Zmat(self.options)
        zmat_obj.run()

        np.set_printoptions(edgeitems=60,linewidth=1000)
        
        # Compute the initial s-vectors
        s_vec = SVectors(
            zmat_obj, self.options, zmat_obj.variable_dictionary_init
        )
        if len(np.shape(self.Proj)) > 2:
            print('this is proj that has been manually sorted by symmetry irrep')
        else:
            print('this is proj, check for this when redundants executed')
            print(self.Proj)
        
        s_vec.run(zmat_obj.cartesians_init, True, proj=self.Proj, second_order=self.options.second_order)
                
        TED_obj = TED(s_vec.proj, zmat_obj)
        print("TED PROJ:")
        print(TED_obj.proj)

        g_mat = GMatrix(zmat_obj, s_vec, self.options)
        g_mat.run()
        
        G = g_mat.G.copy()
        
        if os.path.exists(rootdir + "/fc.grad"):
            print('FC GRAD EXISTS')
            # raise RuntimeError
            g_read_obj = GrRead("fc.grad")
            g_read_obj.run(zmat_obj.cartesians_init)
            print(zmat_obj.cartesians_init)

        # Auto sym cart disps test here:

        # indices = np.triu_indices(len(zmat_obj.cartesians_init.flatten()))
        # indices = np.array(indices).T
        # # print("Coordinate type is:")
        # # print(self.coord_type_init)
        # eigs_init = np.eye(len(s_vec.proj.T))
        # init_disp = TransfDisp(
            # s_vec,
            # zmat_obj,
            # options.disp,
            # eigs_init,
            # True,
            # self.options.disp_tol,
            # TED_obj,
            # self.options,
            # indices,
            # deriv_level = self.options.deriv_level,
            # coord_type = self.coord_type_init
        # )
        # # init_disp.run()
        # A = init_disp.compute_A(
            # s_vec.B, TED_obj.proj, np.eye(len(TED_obj.proj.T)), zmat_obj.mass_weight
        # )
        # A[np.abs(A) < 1e-9] = 0.0
        # print("A: ")
        # print(A.shape)
        # print(A)
        # for i in A:
            # print(np.dot(i,i))
        # print("Transpose:")
        # for i in A.T:
            # print(np.dot(i,i))
        # raise RuntimeError

        init_bool = False
        if os.path.exists(rootdir + "/fc.dat"):
            f_read_obj = FcRead("fc.dat")
        elif os.path.exists(rootdir + "/FCMFINAL"):
            f_read_obj = FcRead("FCMFINAL")
        else:
            init_bool = True
            if cma1_coord == None:
                print("You need to specify the cma1_coord variable for this feature. Check execMerger.run()")
                raise RuntimeError

            os.chdir(os.getcwd() + self.cma1_path)
            if os.path.exists(os.getcwd() + "/fc_int_"+cma1_coord+".dat") and not self.options.second_order:
                if os.path.exists(os.getcwd()+'/DispsInit'):
                    shutil.rmtree("DispsInit")
                f_read_obj = FcRead("fc_int_"+cma1_coord+".dat")
                f_read_obj.run()
                fc_init = ForceConstant(
                    None,
                    [],
                    [],
                    0,
                    self.options,
                    [],
                    [],
                )
                fc_init.FC =  f_read_obj.fc_mat
                os.chdir('..')
                os.chdir('..')
            elif os.path.exists(os.getcwd() + "/fc_cart.dat") and self.options.second_order:
                if os.path.exists(os.getcwd()+'/DispsInit'):
                    shutil.rmtree("DispsInit")
                f_read_obj = FcRead("fc_cart.dat")
                f_read_obj.run()
                fc_init = ForceConstant(
                    None,
                    [],
                    [],
                    0,
                    self.options,
                    [],
                    [],
                )
                fc_init.FC =  f_read_obj.fc_mat
                os.chdir('..')
                os.chdir('..')
            else:
                # First generate displacements in internal coordinates
                eigs_init = np.eye(len(s_vec.proj.T))
                if not self.options.deriv_level:
                    indices = np.triu_indices(len(s_vec.proj.T))
                    indices = np.array(indices).T
                    print("symmetric displacements:")
                    if len(sym_sort) > 1:
                        sym_disps = []
                        for i in sym_sort:
                            for j in indices:
                                if j[0] in i and j[1] in i:
                                    sym_disps.append([j[0],j[1]])
                                    # if j[1] in i:
                                        # sym_disps.append([j[0],j[1]])
                        indices = sym_disps
                
                else:
                    indices = np.arange(len(eigs_init))
                if self.options.second_order:
                    indices = np.triu_indices(len(zmat_obj.cartesians_init.flatten()))
                    indices = np.array(indices).T
                init_disp = TransfDisp(
                    s_vec,
                    zmat_obj,
                    self.options.disp,
                    eigs_init,
                    True,
                    self.options.disp_tol,
                    TED_obj,
                    self.options,
                    indices,
                    deriv_level = self.options.deriv_level,
                    coord_type = self.coord_type_init
                )
                init_disp.run()
                # raise RuntimeError
                prog_init = self.options.program_init
                prog_name_init = prog_init.split("@")[0]
                
                # self.options.calc_init = False
                # print("CALC INIT")
                # print(self.options.calc_init)
                if self.options.calc_init:
                    if os.path.exists(os.getcwd()+'/DispsInit'):
                        shutil.rmtree("DispsInit")
                    dir_obj_init = DirectoryTree(
                        prog_name_init,
                        zmat_obj,
                        init_disp,
                        self.options.cart_insert_init,
                        init_disp.p_disp,
                        init_disp.m_disp,
                        self.options,
                        indices,
                        "templateInit.dat",
                        "DispsInit",
                        deriv_level = self.options.deriv_level
                    )
                    dir_obj_init.run()
                    # raise RuntimeError
                    disp_list = []
                    for i in os.listdir(os.getcwd()):
                        disp_list.append(i)

                    if self.options.cluster != "sapelo":
                        v_template = VulcanTemplate(
                            self.options, len(disp_list), prog_name_init, prog_init
                        )
                        out = v_template.run()
                        with open("displacements.sh", "w") as file:
                            file.write(out)

                        # Submits an array, then checks if all jobs have finished every
                        # 10 seconds.
                        sub = Submit(disp_list,self.options)
                        sub.run()
                    else:
                        s_template = SapeloTemplate(
                            self.options, len(disp_list), prog_name_init, prog_init
                        )
                        out = s_template.run()
                        with open("optstep.sh", "w") as file:
                            file.write(out)
                        for z in range(0, len(disp_list)):
                            source = os.getcwd() + "/optstep.sh"
                            os.chdir("./" + str(z + 1))
                            destination = os.getcwd()
                            shutil.copy2(source, destination)
                            os.chdir("../")
                        sub = Submit(disp_list, self.options)
                        sub.run()

                reap_obj_init = Reap(
                    self.options,
                    eigs_init,
                    indices,
                    self.options.energy_regex_init,
                    self.options.gradient_regex,
                    self.options.success_regex_init,
                    deriv_level = self.options.deriv_level
                )
                reap_obj_init.energy_regex = energy_regex
                reap_obj_init.success_regex = success_regex
                if self.options.calc_init:
                    reap_obj_init.run()
                    os.chdir("..")
                else:
                    os.chdir("DispsInit")
                    reap_obj_init.run()
                    os.chdir("..")

                # nate
                if not self.options.deriv_level:
                    p_array_init = reap_obj_init.p_en_array
                    m_array_init = reap_obj_init.m_en_array
                    ref_en_init = reap_obj_init.ref_en
                else:
                    cart_p_array_init = reap_obj_init.p_grad_array
                    cart_m_array_init = reap_obj_init.m_grad_array
                    p_array_init = np.zeros(np.eye(len(eigs_init)).shape)
                    m_array_init = np.zeros(np.eye(len(eigs_init)).shape)
                    ref_en_init = None
                    # Need to convert this array here from cartesians to internals using projected A-tensor
                    for i in indices:
                        grad_s_vec = SVectors(
                            zmat_obj, self.options, zmat_obj.variable_dictionary_init
                        )
                        grad_s_vec.run(init_disp.p_disp[i],False)
                        A_proj = np.dot(LA.pinv(grad_s_vec.B),TED_obj.proj)
                        p_array_init[i] = np.dot(cart_p_array_init[i].T,A_proj)
                        grad_s_vec.run(init_disp.m_disp[i],False)
                        A_proj = np.dot(LA.pinv(grad_s_vec.B),TED_obj.proj)
                        m_array_init[i] = np.dot(cart_m_array_init[i].T,A_proj)
                    

                fc_init = ForceConstant(
                    init_disp,
                    p_array_init,
                    m_array_init,
                    ref_en_init,
                    self.options,
                    indices,
                    reap_obj_init,
                    deriv_level=self.options.deriv_level,
                    coord_type_init=coord_type_init
                )
                fc_init.run()
                print("Computed Force Constants:")
                print(fc_init.FC)
                if self.options.second_order:
                    p_array_grad = np.array([])
                    m_array_grad = np.array([])
                    for i in range(len(p_array_init)):
                        p_array_grad = np.append(p_array_grad,p_array_init[i,i])
                        m_array_grad = np.append(m_array_grad,m_array_init[i,i])
                    grad_init = ForceConstant(
                        init_disp,
                        p_array_grad,
                        m_array_grad,
                        ref_en_init,
                        self.options,
                        indices,
                        reap_obj_init,
                        deriv_level=1
                    )
                    grad_init.run()
                    print("Computed Gradient:")
                    print(grad_init.FC)
                f_conv_obj = FcConv(
                    fc_init.FC,
                    s_vec,
                    zmat_obj,
                    "internal",
                    False,
                    TED_obj,
                    self.options.units,
                    self.options.second_order
                )
                f_conv_obj.N = len(fc_init.FC)
                if self.coord_type_init == "internal":
                    f_conv_obj.print_const(fc_name="fc_int_"+cma1_coord+".dat")
                    shutil.move(os.getcwd() + "/fc_int_"+cma1_coord+".dat", os.getcwd()+"/.." + self.cma1_path +"/fc_int_"+cma1_coord+".dat")
                elif self.coord_type_init == "cartesian":
                    f_conv_obj.print_const(fc_name="fc_cart.dat")
                    shutil.move(os.getcwd() + "/fc_cart.dat", os.getcwd()+"/.." + self.cma1_path +"/fc_cart.dat")
                print("Force Constants saved at:")
                print(self.cma1_path)
                if self.options.second_order:
                    fc_name = "fc_cart.grad"
                    fc_output = ""
                    g_print = grad_init.FC.copy()
                    g_print = g_print.flatten()
                    for i in range(len(g_print) // 3):
                        fc_output += "{:20.10f}".format(g_print[3 * i])
                        fc_output += "{:20.10f}".format(g_print[3 * i + 1])
                        fc_output += "{:20.10f}".format(g_print[3 * i + 2])
                        fc_output += "\n"
                    if len(g_print) % 3:
                        for i in range(len(g_print) % 3):
                            fc_output += "{:20.10f}".format(
                                g_print[3 * (len(g_print) // 3) + i]
                            )
                        fc_output += "\n"
                    with open(fc_name, "w+") as file:
                        file.write(fc_output)
                    # f_conv_obj.print_const(fc_name="fc_cart.grad")
                    shutil.move(os.getcwd() + "/fc_cart.grad", os.getcwd()+"/.." + self.cma1_path +"/fc_cart.grad")
                    print("Gradient saved at:")
                    print(self.cma1_path)
                    f_conv_obj.run(grad=grad_init.FC)
                    g_read_obj = GrRead("fc.grad")
                    g_read_obj.cart_grad = grad_init.FC

                os.chdir("..")
        
        if not init_bool:
            f_read_obj.run()
            f_conv_obj = FcConv(
                f_read_obj.fc_mat,
                s_vec,
                zmat_obj,
                "internal",
                False,
                TED_obj,
                self.options.units,
                self.options.second_order
            )
            if self.options.second_order:
                f_conv_obj.run(grad=g_read_obj.cart_grad)
                # raise RuntimeError
            else:
                f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_init.FC
            if self.options.second_order:
                F = f_conv_obj.F
                F = np.dot(TED_obj.proj.T, np.dot(F, TED_obj.proj))
        
        self.options.deriv_level = 0
        
        if self.options.coords != "ZMAT" and not init_bool:
            F = np.dot(TED_obj.proj.T, np.dot(F, TED_obj.proj))
            # F[np.abs(F) < 1.0e-6] = 0
            # print("Nat Int force constants:")
            # print(F)
            if self.options.second_order:
                grad_proj = np.dot(TED_obj.proj.T,f_conv_obj.v_q)
        
        if self.options.coords != "ZMAT":
            g_mat.G = np.dot(TED_obj.proj.T, np.dot(g_mat.G, TED_obj.proj))
        
        TED_obj.run(np.eye(TED_obj.proj.shape[1]),np.zeros(TED_obj.proj.shape[1]))
       

        

        if len(sym_sort) > 1:
            F_sym = F[flat_sym_sort].copy()
            F_sym = F_sym[:,flat_sym_sort]
            print("Sym Force Constants:")
            print(F_sym)
            # F_sym = F_sym[flat_sym_sort.argsort()].copy()
            # F_sym = F_sym[:,flat_sym_sort.argsort()]
            
            g_sym = g_mat.G[flat_sym_sort].copy()
            g_sym = g_sym[:,flat_sym_sort]
            g_sym[np.abs(g_sym) < 1e-9] = 0
            print("Sym G-Matrix:")
            print(sym_sort)
            print(g_sym)

        print("Initial Force Constants:")
        print(F.shape)
        print(F)
        
        print("Initial G-Matrix:")
        g_mat.G[np.abs(g_mat.G) < 1e-9] = 0
        print(g_mat.G)

        print("Initial Frequencies:")
        init_GF = GFMethod(
            g_mat.G.copy(),
            F.copy(),
            self.options.tol,
            self.options.proj_tol,
            zmat_obj,
            TED_obj,
            False
        )
        init_GF.run()
       
        
        # print("TED for sym purposes: ")
        init_GF.ted.TED[np.abs(init_GF.ted.TED) < 1e-5] = 0
        print(init_GF.ted.TED)
        # raise RuntimeError
        
        self.ref_init = init_GF.freq
        if len(sym_sort):
            self.irreps_init,flat_sym_freqs = self.mode_symmetry_sort(init_GF.ted.TED,sym_sort,self.ref_init)
            self.ref_init = np.array(flat_sym_freqs)

        # Now for the TED check.
        G = np.dot(np.dot(LA.inv(init_GF.L), g_mat.G), LA.inv(init_GF.L).T)
        G[np.abs(G) < self.options.tol] = 0
        F = np.dot(np.dot(init_GF.L.T, F), init_GF.L)
        F[np.abs(F) < self.options.tol] = 0
        # if self.options.second_order:
            # grad_n = np.dot(init_GF.L.T, grad_proj)
            # print("Normal Mode Gradients:")
            # for i in range(len(grad_n)):
                # print(str(i+1) + ": " + str(grad_n[i]))
        
        print("TED Frequencies:")
        TED_GF = GFMethod(
            G,
            F,
            self.options.tol,
            self.options.proj_tol,
            zmat_obj,
            TED_obj,
            False
        )
        TED_GF.run()
       
        proj_tol = 1.0e-3
        eig_inv = inv(init_GF.L)  # (Normal modes (Q) x Sym internals (S) )
        for i in range(len(eig_inv)):
            eig_inv[i] = eig_inv[i] / LA.norm(eig_inv[i])
            eig_inv[i][
                np.abs(eig_inv[i]) < np.max(np.abs(eig_inv[i])) * proj_tol
            ] = 0
        
        # Now run the TZ force constant transformation
        zmat_obj2 = Zmat(self.options)
        zmat_obj2.run(zmat_name="zmat2")
        
        self.options.man_proj = True
        
        
        s_vec = SVectors(
            zmat_obj2, self.options, zmat_obj2.variable_dictionary_init
        )
        s_vec.run(zmat_obj2.cartesians_init, True, proj=TED_obj.proj)
                
        TED_obj = TED(s_vec.proj, zmat_obj2)
                
        g_mat = GMatrix(zmat_obj2, s_vec, self.options)
        g_mat.run()
        
        
        G = g_mat.G.copy()
        Gtz = G.copy()
        init_bool = False
        if os.path.exists(rootdir + "/fc2.dat"):
            f_read_obj = FcRead("fc2.dat")
        elif os.path.exists(rootdir + "/FCMFINAL2"):
            f_read_obj = FcRead("FCMFINAL2")
        else:
            raise RuntimeError
        
        # check this, this seems like an unnecessary else statement.
        if not init_bool:
            f_read_obj.run()
            f_conv_obj = FcConv(
                f_read_obj.fc_mat,
                s_vec,
                zmat_obj2,
                "internal",
                False,
                TED_obj,
                self.options.units,
                False,
            )
            f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_init.FC
        
        
        
        # redundant basis 
        G = np.dot(np.dot(TED_obj.proj.T,G),TED_obj.proj)
        print("Giraffe G")
        # G[np.abs(G) < 1.0e-9] = 0 
        # print(G)
        # if len(sym_sort) > 1:
            # print(g_sym)
        G = np.dot(np.dot(eig_inv, G), eig_inv.T)
        # G[np.abs(G) < 1.0e-7] = 0 
        # print(G[34])
        # Conversion to aJ/Ang
        F_aJ = F.copy()
        F_aJ *= 4.3597447222071
        F_aJ /= 0.529177210903
        F = np.dot(np.dot(TED_obj.proj.T,F),TED_obj.proj)
        F_aJ = np.dot(np.dot(TED_obj.proj.T,F_aJ),TED_obj.proj)
        # if len(sym_sort) > 1:
            # F = F[flat_sym_sort]
            # F = F[:,flat_sym_sort]
            # F_aJ = F_aJ[flat_sym_sort]
            # F_aJ = F_aJ[:,flat_sym_sort]
        print("Giraffe F")
        # F[np.abs(F) < 1.0e-5] = 0 
        # print(F)
        # if len(sym_sort) > 1:
            # print(sym_sort)
            # F_sym = F[flat_sym_sort]
            # F_sym = F_sym[:,flat_sym_sort]
            # print(F_sym)
        # print("aJ F")
        # F_aJ[np.abs(F) < 1.0e-5] = 0 
        # print(F_aJ)
        # if len(sym_sort) > 1:
            # print("aJ/A Sym Force Constants:")
            # print(flat_sym_sort+1)
            # F_aJ = F_aJ[flat_sym_sort]
            # F_aJ = F_aJ[:,flat_sym_sort]
            # print(F_aJ)
        F = np.dot(np.dot(inv(eig_inv).T, F), inv(eig_inv))
        # F[np.abs(F) < 1.0e-5] = 0 
        print(F)
        # F[np.abs(F) < self.options.tol] = 0
         
        # print('Testing G:')
        # print(G)
        # print('Testing F:')
        # print(F)
        full_GF = GFMethod(
            G,
            F,
            self.options.tol,
            self.options.proj_tol,
            zmat_obj2,
            TED_obj,
            False
        )
        full_GF.run()
        self.ted = full_GF.ted.TED      # TED matrix
        
        # Print Full TED here in projected basis

        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" Full Hessian TED"))
        print("////////////////////////////////////////////")
        TED_obj.run(np.dot(init_GF.L, full_GF.L), full_GF.freq, rect_print=False)
        
        self.reference_freq = full_GF.freq 
        if len(sym_sort):
            self.irreps_ref,flat_sym_freqs = self.mode_symmetry_sort(TED_obj.TED,sym_sort,self.reference_freq)
            self.reference_freq = np.array(flat_sym_freqs)
        
        m = 2 
        var = 0.95 
        
        
        self.reference_TED = TED_obj.TED
        ref_TED = self.reference_TED

        # raise RuntimeError
        if self.options.coords == 'Redundants':
            L_B = full_GF.L
        elif self.options.coords == 'Custom':
            L_A = full_GF.L
         
        
        def n_largest(n, FC):
            indexes = []
            upper_triang = abs(np.triu(FC,n))
            for i in range(0,n):
                fc_cma2 = np.where(upper_triang == upper_triang.max())
                index = [fc_cma2[0][0], fc_cma2[1][0]]
                indexes.append(index)
                upper_triang[index[0],index[1]] = 0
            print(indexes)
            return indexes
        
        np.set_printoptions(edgeitems=60,linewidth=10000)
        print("Full Force constant matrix in lower level normal mode basis:")
        print(F)
        if self.options.coords == 'Redundant':
            self.F_redundant = F
        elif self.options.coords == 'Custom':
            self.F_custom = F
        elif self.options.coords == 'ZMAT' :
            self.F_zmat = F
        else:
            pass
        Fdiag = copy.copy(np.diag(np.diag(F)))
        
        print("Diagonal Force constant matrix in lower level normal mode basis:")
        print(Fdiag)
        diag_GF = GFMethod(
            G,
            Fdiag,
            self.options.tol,
            self.options.proj_tol,
            zmat_obj2,
            TED_obj,
            False
        )
        
        diag_GF.run()

        self.Freq_CMA0 = diag_GF.freq

        # if self.options.coords == 'Redundant':
            # self.Freq_redundant = diag_GF.freq
        # elif self.options.coords == 'Custom':
            # self.Freq_custom = diag_GF.freq
        # elif self.options.coords == 'ZMAT' :
            # self.Freq_zmat = diag_GF.freq
        # else:
            # pass
        

        # Print Diagonal TED here in projected basis

        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" CMA-0 TED"))
        print("////////////////////////////////////////////")
        TED_obj.run(np.dot(init_GF.L, diag_GF.L), diag_GF.freq, rect_print=False)
        
        if len(sym_sort):
            self.irreps_CMA0,flat_sym_freqs = self.mode_symmetry_sort(TED_obj.TED,sym_sort,self.Freq_CMA0)
            self.Freq_CMA0 = np.array(flat_sym_freqs)

        # Beginning of the condensed, new off-diag code.
        if self.options.off_diag:
            # od_inds = self.od_inds
            temp = copy.copy(Fdiag)
            if self.options.off_diag == 1:
                print("Adding on these off-diagonals:")
                print(od_inds)
                for od_ind in od_inds:
                    element = F[od_ind[0], od_ind[1]] 
                    temp[od_ind[0], od_ind[1]] = element
                    temp[od_ind[1], od_ind[0]] = element
                print('Time for some off-diags')
                cma1_GF = GFMethod(
                    G,
                    temp,
                    self.options.tol,
                    self.options.proj_tol,
                    zmat_obj2,
                    TED_obj,
                    False
                )
                cma1_GF.run()
                cma1_Freq = cma1_GF.freq.copy()

                
                print("////////////////////////////////////////////")
                print("//{:^40s}//".format(" CMA-1 TED"))
                print("////////////////////////////////////////////")
                TED_obj.run(np.dot(init_GF.L, cma1_GF.L), cma1_GF.freq, rect_print=False)
                
                if len(sym_sort):
                    self.irreps_CMA1,flat_sym_freqs = self.mode_symmetry_sort(TED_obj.TED,sym_sort,cma1_Freq)
                    cma1_Freq = np.array(flat_sym_freqs)
                
                # self.RMSD = np.append(self.RMSD,cma1_rmsd)
                self.Freq_cma1 = cma1_Freq

            elif self.options.off_diag == 2:
                self.Freq_cma2 = np.array([])
                self.eta_num = np.array([])
                self.eta_denom = np.array([])
                self.total_off_diags = np.array([])
                
                for xi_tol_i in xi_tol:
                    print(xi_tol_i)
                    print('Time for some off-diags')
                    if len(self.options.other_F_matrix) and os.path.exists(os.getcwd() + "/inter_fc.dat"):
                        f_read_obj_inter = FcRead("inter_fc.dat")
                        f_read_obj_inter.run()
                        F_inter = f_read_obj_inter.fc_mat
                        F_inter = np.dot(np.dot(inv(eig_inv).T, F_inter), inv(eig_inv))
                        print("F_inter:")
                        print(F_inter)
                        print("F_A:")
                        print(F)
                        xi = F_inter * 0
                        od_inds = []
                        if len(sym_sort) > 1:
                            self.total_off_diags_buff = 0
                            for irrep in self.irreps_init:
                                if len(irrep) > 1:
                                    for i in range(len(irrep)):
                                        for j in range(i):
                                            if i != j:
                                                a = irrep[i]
                                                b = irrep[j]
                                                buff = np.abs(F_inter[a,b])
                                                xi[a,b] = buff / np.sqrt(np.abs(F_inter[a,a])*np.abs(F_inter[b,b]))
                                                if xi[a,b] > xi_tol_i:
                                                    od_inds.append([a,b])
                                    self.total_off_diags_buff += (len(irrep)**2 - len(irrep))/2
                        else:
                            self.total_off_diags_buff = (len(xi)**2 - len(xi))/2
                            for i in range(len(xi)):
                                for j in range(i+1):
                                    if i != j:
                                        buff = np.abs(F_inter[i,j])
                                        xi[i,j] = buff / np.sqrt(np.abs(F_inter[i,i])*np.abs(F_inter[j,j]))
                                        if xi[i,j] > xi_tol_i:
                                            od_inds.append([i,j])
                        
                        print("CMA2 off-diagonal elements:")
                        print(od_inds)
                        self.cma_off_diags = len(od_inds)
                        # self.off_diags = np.append(self.off_diags,len(od_inds))
                        self.total_off_diags = np.append(self.total_off_diags,self.total_off_diags_buff)
                        # self.perc_off_diags = (self.cma_off_diags / self.total_off_diags) * 100
                        self.eta_num = np.append(self.eta_num,self.cma_off_diags*1.0)
                        self.eta_denom = np.append(self.eta_denom,len(self.Freq_CMA0)*1.0)
                        # raise RuntimeError
                        # extras = [[0,1],[0,2],[1,2]]
                        # print('extras')
                        # print(len(extras))
                        # print(extras)
                        print("temp:")
                        print(temp)
                        for od_ind in od_inds:
                            print(od_ind[0], od_ind[1])
                            element = F[od_ind[0], od_ind[1]]
                            print(element)
                            temp[od_ind[0], od_ind[1]] = element
                            temp[od_ind[1], od_ind[0]] = element
                    print(temp)
                    print(F)
                    cma2_GF = GFMethod(
                        G,
                        temp,
                        self.options.tol,
                        self.options.proj_tol,
                        zmat_obj2,
                        TED_obj,
                        False
                    )
                    cma2_GF.run()
                    cma2_Freq = cma2_GF.freq.copy()

                    
                    print("////////////////////////////////////////////")
                    print("//{:^40s}//".format(" CMA-2 TED"))
                    print("////////////////////////////////////////////")
                    TED_obj.run(np.dot(init_GF.L, cma2_GF.L), cma2_GF.freq, rect_print=False)
                    
                    if len(sym_sort):
                        self.irreps_CMA2,flat_sym_freqs = self.mode_symmetry_sort(TED_obj.TED,sym_sort,cma2_Freq)
                        cma2_Freq = np.array(flat_sym_freqs)
                    
                    # self.RMSD = np.append(self.RMSD,cma2_rmsd)
                    self.Freq_cma2 = np.append(self.Freq_cma2,cma2_Freq,axis=0)
                self.Freq_cma2 = np.reshape(self.Freq_cma2,(len(xi_tol),-1))
            else:
                print("Only CMA-1 and CMA-2 off_diag algorithms are implemented at the moment.")
                print("Please enter either 1 or 2 for the off_diag option.")
                raise RuntimeError

        # if self.options.n_cma2 > 0:
            # self.Freq_cma2 = np.array([])
            # # self.cma2_rmsd = np.array([])
            # self.eta_num = np.array([])
            # self.eta_denom = np.array([])
            # self.total_off_diags = np.array([])
            # for xi_tol_i in xi_tol:
                # if self.options.off_diag:
                    # algo = Algorithm(eigs, None, self.options)
                    # algo.run()
                    # temp = np.zeros((eigs,eigs))  
                    # for z, extra in enumerate(algo.indices):
                        # element = F[extra[0], extra[1]] 
                        # temp[extra[0], extra[1]] = element
                        # temp[extra[1], extra[0]] = element
                # else:
                    # # extras = n_largest(self.options.n_cma2, np.abs(copy.copy(F)))
                    # extras = []
                    # extras = [[7,8]]
                    # # print("Inter off diag indices:")
                    # # for i in range(len(Fdiag)):
                        # # for j in range(len(Fdiag)-i-1):
                            # # # print([i,j+i+1])
                            # # extras.append([i,j+i+1])

                    # # raise RuntimeError
                    # temp = copy.copy(Fdiag)
                    # if len(self.options.other_F_matrix) and os.path.exists(os.getcwd() + "/inter_fc.dat"):
                        # f_read_obj_inter = FcRead("inter_fc.dat")
                        # f_read_obj_inter.run()
                        # F_inter = f_read_obj_inter.fc_mat
                        # F_inter = np.dot(np.dot(inv(eig_inv).T, F_inter), inv(eig_inv))
                        # print("F_inter:")
                        # print(F_inter)
                        # print("F_A:")
                        # print(F)
                        # xi = F_inter * 0
                        # extras = []
                        # if len(sym_sort) > 1:
                            # self.total_off_diags_buff = 0
                            # for irrep in self.irreps_init:
                                # if len(irrep) > 1:
                                    # for i in range(len(irrep)):
                                        # for j in range(i):
                                            # if i != j:
                                                # a = irrep[i]
                                                # b = irrep[j]
                                                # buff = np.abs(F_inter[a,b])
                                                # xi[a,b] = buff / np.sqrt(np.abs(F_inter[a,a])*np.abs(F_inter[b,b]))
                                                # if xi[a,b] > xi_tol_i:
                                                    # extras.append([a,b])
                                    # self.total_off_diags_buff += (len(irrep)**2 - len(irrep))/2
                        # else:
                            # self.total_off_diags_buff = (len(xi)**2 - len(xi))/2
                            # for i in range(len(xi)):
                                # for j in range(i+1):
                                    # if i != j:
                                        # buff = np.abs(F_inter[i,j])
                                        # xi[i,j] = buff / np.sqrt(np.abs(F_inter[i,i])*np.abs(F_inter[j,j]))
                                        # if xi[i,j] > xi_tol_i:
                                            # extras.append([i,j])
                        
                        # self.cma_off_diags = len(extras)
                        # # self.off_diags = np.append(self.off_diags,len(extras))
                        # self.total_off_diags = np.append(self.total_off_diags,self.total_off_diags_buff)
                        # # self.perc_off_diags = (self.cma_off_diags / self.total_off_diags) * 100
                        # self.eta_num = np.append(self.eta_num,self.cma_off_diags*1.0)
                        # self.eta_denom = np.append(self.eta_denom,len(self.Freq_CMA0)*1.0)
                        # # raise RuntimeError
                        # # extras = [[0,1],[0,2],[1,2]]
                        # # print('extras')
                        # # print(len(extras))
                        # # print(extras)
                        # for z, extra in enumerate(extras):
                            # element = F[extra[0], extra[1]] 
                            # temp[extra[0], extra[1]] = element
                            # temp[extra[1], extra[0]] = element
                        

                    # if len(self.options.other_F_matrix) and os.path.exists(os.getcwd() + "/inter_fc.dat"):
                        # pass
                        # # print(os.getcwd())
                        # # f_read_obj_inter = FcRead("inter_fc.dat")
                        # # f_read_obj_inter.run()
                        # # F_inter = f_read_obj_inter.fc_mat
                        # # F_inter = np.dot(np.dot(inv(eig_inv).T, F_inter), inv(eig_inv))
                        # # print("F_inter:")
                        # # print(F_inter)
                        # # print("F_A:")
                        # # print(F)
                        # # # raise RuntimeError
                        # # for z, extra in enumerate(extras):
                            # # element = F_inter[extra[0], extra[1]] 
                            # # temp[extra[0], extra[1]] = element
                            # # temp[extra[1], extra[0]] = element
                    # else:
                        # for z, extra in enumerate(extras):
                            # element = F[extra[0], extra[1]] 
                            # temp[extra[0], extra[1]] = element
                            # temp[extra[1], extra[0]] = element
                    # print('CMA2 FC matrix')
                    # print(temp) 
                
                
                # print('Time for some off-diags')
                # cma2_GF = GFMethod(
                    # G,
                    # temp,
                    # self.options.tol,
                    # self.options.proj_tol,
                    # zmat_obj2,
                    # TED_obj,
                    # False
                # )
                # cma2_GF.run()
                # cma2_Freq = cma2_GF.freq.copy()

                
                # print("////////////////////////////////////////////")
                # print("//{:^40s}//".format(" CMA-2 TED"))
                # print("////////////////////////////////////////////")
                # TED_obj.run(np.dot(init_GF.L, cma2_GF.L), cma2_GF.freq, rect_print=False)
                
                # if len(sym_sort):
                    # self.irreps_CMA2,flat_sym_freqs = self.mode_symmetry_sort(TED_obj.TED,sym_sort,cma2_Freq)
                    # cma2_Freq = np.array(flat_sym_freqs)
                
                # # self.RMSD = np.append(self.RMSD,cma2_rmsd)
                # self.Freq_cma2 = np.append(self.Freq_cma2,cma2_Freq,axis=0)
            
            # if not len(xi_tol):
                # print("We entered the if")
                # temp = copy.copy(Fdiag)
                # extras = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,5]]
                # print('extras')
                # print(len(extras))
                # print(extras)
                # for z, extra in enumerate(extras):
                    # element = F[extra[0], extra[1]] 
                    # temp[extra[0], extra[1]] = element
                    # temp[extra[1], extra[0]] = element
                # print('CMA1 FC matrix')
                # print(temp) 
                
                # print('Time for some off-diags')
                # cma2_GF = GFMethod(
                    # G,
                    # temp,
                    # self.options.tol,
                    # self.options.proj_tol,
                    # zmat_obj2,
                    # TED_obj,
                    # False
                # )
                # cma2_GF.run()
                # cma2_Freq = cma2_GF.freq.copy()

                
                # print("////////////////////////////////////////////")
                # print("//{:^40s}//".format(" CMA-2 TED"))
                # print("////////////////////////////////////////////")
                # TED_obj.run(np.dot(init_GF.L, cma2_GF.L), cma2_GF.freq, rect_print=False)
                
                # if len(sym_sort):
                    # self.irreps_CMA2,flat_sym_freqs = self.mode_symmetry_sort(TED_obj.TED,sym_sort,cma2_Freq)
                    # cma2_Freq = np.array(flat_sym_freqs)
                
                # # self.RMSD = np.append(self.RMSD,cma2_rmsd)
                # self.Freq_cma2 = cma2_Freq
            
            # print(self.Freq_cma2)
            # if len(xi_tol):
                # self.Freq_cma2 = np.reshape(self.Freq_cma2,(len(xi_tol),-1))
        
        def n_largest(n, FC):
            indexes = []
            upper_triang = abs(np.triu(FC,1))
            length = len(upper_triang)
            for i in range(0,n):
                index = np.argmax(upper_triang)
                if index > length:
                    two_d = [index // length, index % length]
                else:
                    two_d = [0,index]
                indexes.append(two_d)
                
                upper_triang[two_d[0],two_d[1]] = 0
            return indexes
        
    # This function returns a sorted list of frequencies, with any non-coupling frequencies being deleted.
    def mode_symmetry_sort(self,TED,sym_sort,freqs):
        ref_TED_init = TED
        sym_modes = []
        for irrep in sym_sort:
            irrep_modes = []
            for i in range(len(ref_TED_init.T)):
                Sum = 0
                for j in irrep:
                    Sum += ref_TED_init.T[i,j]
                # print(i)
                # print(irrep)
                # print(Sum)
                if Sum > 90.:
                    irrep_modes.append(i)
            # print(np.array(irrep)+1)
            # print(np.array(irrep_modes)+1)
            if len(irrep_modes) != len(irrep):
                print("Something's wrong with the irrep symmetry sorter:")
                raise RuntimeError
            sym_modes.append(irrep_modes)

        sym_freqs = copy.deepcopy(sym_modes)
        del_list = []
        for i in range(len(sym_modes)):
            if len(sym_modes[i]) > 1:
                for j in range(len(sym_modes[i])):
                    index = sym_modes[i][j]
                    sym_freqs[i][j] = freqs[index].copy()
                sym_freqs[i].reverse()
            elif len(sym_modes[i]) == 1:
                del_list.append(i)
            else:
                pass
        del_list.reverse()
        if len(del_list):
            for i in del_list:
                print(freqs[sym_modes[i][0]])
        for i in del_list:
            del sym_freqs[i]
        flat_sym_freqs = [
            x
            for xs in sym_freqs
            for x in xs
        ]
        flat_sym_freqs = np.array(flat_sym_freqs)

        return sym_modes, flat_sym_freqs
