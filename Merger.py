import os
import re
import sys
import shutil
import json
import subprocess
import time
import numpy as np

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
        #print("nothing to init")
        options_kwargs = {
            "queue": "gen4.q,gen6.q,gen5.q",
            "program": "molpro@2010.1.67+mpi",
            "energy_regex": r"\(T\) energy\s+(\-\d+\.\d+)",
            'energy_regex' : r"\s*\!CCSD\(T\) total energy\s+(-\d+\.\d+)",
            "cart_insert": 9,
            "calc" : False,
            "calc_init" : False,
            "success_regex": r"Variable memory released",
            # "reduced_disp" : True,
            # "disp" : 1.0
        }
        options_obj = Options(**options_kwargs)
        self.options = options_obj 
        self.cma1_path = cma1_path
    #function that returns diagonal fc matrix + n-largest off-diagonal elements
    def run(self, opts, Proj, energy_regex=None, success_regex=None, cma1_coord=None, sym_sort=None):
    # def run(self, opts, Proj, energy_regex=None, success_regex=None, cma1_coord=None):

        print("You have imported the merger script!")
        
        self.Proj = Proj 
        options = opts 
        #options = options_obj
        # options.cart_insert_init = 26
        options.cart_insert_init = 9
        rootdir = os.getcwd()
        zmat_obj = Zmat(options)
        zmat_obj.run()

        np.set_printoptions(edgeitems=60,linewidth=1000)
        
        # Compute the initial s-vectors
        s_vec = SVectors(
            zmat_obj, options, zmat_obj.variable_dictionary_init
        )
        if len(np.shape(self.Proj)) > 2:
            print('this is proj that has been manually sorted by symmetry irrep')
        else:
            print('this is proj, check for this when redundants executed')
            print(self.Proj)
        s_vec.run(zmat_obj.cartesians_init, True, proj=self.Proj, second_order=options.second_order)
                
        TED_obj = TED(s_vec.proj, zmat_obj)
        print("TED PROJ:")
        print(TED_obj.proj)
        
        g_mat = GMatrix(zmat_obj, s_vec, options)
        g_mat.run()
        
        G = g_mat.G.copy()
        
        if os.path.exists(rootdir + "/fc.grad"):
            print('FC GRAD EXISTS')
            # raise RuntimeError
            g_read_obj = GrRead("fc.grad")
            g_read_obj.run(zmat_obj.cartesians_init)


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
            if os.path.exists(os.getcwd() + "/fc_int_"+cma1_coord+".dat"):
                if os.path.exists(os.getcwd()+'/DispsInit'):
                    shutil.rmtree("DispsInit")
                f_read_obj = FcRead("fc_int_"+cma1_coord+".dat")
                f_read_obj.run()
                fc_init = ForceConstant(
                    None,
                    [],
                    [],
                    0,
                    options,
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
                else:
                    indices = np.arange(len(eigs_init))
                
                init_disp = TransfDisp(
                    s_vec,
                    zmat_obj,
                    options.disp,
                    eigs_init,
                    True,
                    options.disp_tol,
                    TED_obj,
                    options,
                    indices,
                    deriv_level = self.options.deriv_level
                )
                init_disp.run()
                prog_init = options.program_init
                prog_name_init = prog_init.split("@")[0]
                
                # options.calc_init = False
                if options.calc_init:
                    if os.path.exists(os.getcwd()+'/DispsInit'):
                        shutil.rmtree("DispsInit")
                    dir_obj_init = DirectoryTree(
                        prog_name_init,
                        zmat_obj,
                        init_disp,
                        options.cart_insert_init,
                        init_disp.p_disp,
                        init_disp.m_disp,
                        options,
                        indices,
                        "templateInit.dat",
                        "DispsInit",
                        deriv_level = self.options.deriv_level
                    )
                    dir_obj_init.run()
                    disp_list = []
                    for i in os.listdir(os.getcwd()):
                        disp_list.append(i)

                    if options.cluster != "sapelo":
                        v_template = VulcanTemplate(
                            options, len(disp_list), prog_name_init, prog_init
                        )
                        out = v_template.run()
                        with open("displacements.sh", "w") as file:
                            file.write(out)

                        # Submits an array, then checks if all jobs have finished every
                        # 10 seconds.
                        sub = Submit(disp_list,options)
                        sub.run()
                    else:
                        s_template = SapeloTemplate(
                            options, len(disp_list), prog_name_init, prog_init
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
                        sub = Submit(disp_list, options)
                        sub.run()

                print("deriv_level:")
                print(self.options.deriv_level)
                reap_obj_init = Reap(
                    options,
                    eigs_init,
                    indices,
                    options.gradient_regex,
                    options.energy_regex_init,
                    None,
                    options.success_regex_init,
                    deriv_level = self.options.deriv_level
                )
                reap_obj_init.energy_regex = energy_regex
                reap_obj_init.success_regex = success_regex
                if options.calc_init:
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
                    options,
                    indices,
                    deriv_level=self.options.deriv_level
                )
                fc_init.run()
                print("Computed Force Constants:")
                print(fc_init.FC)
                f_conv_obj = FcConv(
                    fc_init.FC,
                    s_vec,
                    zmat_obj,
                    "internal",
                    False,
                    TED_obj,
                    options.units,
                    False
                )
                f_conv_obj.N = len(fc_init.FC)
                f_conv_obj.print_const(fc_name="fc_int_"+cma1_coord+".dat")
                print("Force Constants saved at:")
                print(self.cma1_path)
                shutil.move(os.getcwd() + "/fc_int_"+cma1_coord+".dat", os.getcwd()+"/.." + self.cma1_path +"/fc_int_"+cma1_coord+".dat")

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
                options.units,
                options.second_order
            )
            if options.second_order:
                f_conv_obj.run(grad=g_read_obj.cart_grad)
            else:
                f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_init.FC
        self.options.deriv_level = 0
        
        if options.coords != "ZMAT" and not init_bool:
            F = np.dot(TED_obj.proj.T, np.dot(F, TED_obj.proj))
        
        if options.coords != "ZMAT":
            g_mat.G = np.dot(TED_obj.proj.T, np.dot(g_mat.G, TED_obj.proj))
        
        TED_obj.run(np.eye(TED_obj.proj.shape[1]),np.zeros(TED_obj.proj.shape[1]))
        
        print("sym_sort:")
        print(sym_sort)
        if len(sym_sort) > 1:
            Fbuff1 = np.array([])
            Fbuff2 = {}
            Gbuff1 = np.array([])
            Gbuff2 = {}
            for i in range(len(sym_sort)):
                Fbuff1 = F.copy()
                Fbuff1 = Fbuff1[sym_sort[i]]
                Fbuff1 = np.array([Fbuff1[:,sym_sort[i]]])
                Fbuff2[str(i)] = Fbuff1.copy()
                Gbuff1 = g_mat.G.copy()
                Gbuff1 = Gbuff1[sym_sort[i]]
                Gbuff1 = np.array([Gbuff1[:,sym_sort[i]]])
                Gbuff2[str(i)] = Gbuff1.copy()
            Fbuff3 = Fbuff2[str(0)][0].copy()
            Gbuff3 = Gbuff2[str(0)][0].copy()
            for i in range(len(sym_sort)-1):
                Fbuff3 = np.block([
                    [Fbuff3,                                        np.zeros((len(Fbuff3),len(Fbuff2[str(i+1)][0])))],
                    [np.zeros((len(Fbuff2[str(i+1)][0]),len(Fbuff3))), Fbuff2[str(i+1)][0]]
                    ])
                Gbuff3 = np.block([
                    [Gbuff3,                                        np.zeros((len(Gbuff3),len(Gbuff2[str(i+1)][0])))],
                    [np.zeros((len(Gbuff2[str(i+1)][0]),len(Gbuff3))), Gbuff2[str(i+1)][0]]
                    ])
            F = Fbuff3
            g_mat.G = Gbuff3

        print("Initial Frequencies:")
        init_GF = GFMethod(
            g_mat.G.copy(),
            F.copy(),
            options.tol,
            options.proj_tol,
            zmat_obj,
            TED_obj,
            False
        )
        init_GF.run()
        self.ref_init = init_GF.freq
        
        # Now for the TED check.
        G = np.dot(np.dot(LA.inv(init_GF.L), g_mat.G), LA.inv(init_GF.L).T)
        G[np.abs(G) < options.tol] = 0
        F = np.dot(np.dot(init_GF.L.T, F), init_GF.L)
        F[np.abs(F) < options.tol] = 0
        
        print("TED Frequencies:")
        TED_GF = GFMethod(
            G,
            F,
            options.tol,
            options.proj_tol,
            zmat_obj,
            TED_obj,
            False
        )
        TED_GF.run()
       
        eigs = len(TED_GF.S)
        print('eigs')
        print(eigs)
        self.eigs = eigs


 
        proj_tol = 1.0e-3
        eig_inv = inv(init_GF.L)  # (Normal modes (Q) x Sym internals (S) )
        for i in range(len(eig_inv)):
            eig_inv[i] = eig_inv[i] / LA.norm(eig_inv[i])
            eig_inv[i][
                np.abs(eig_inv[i]) < np.max(np.abs(eig_inv[i])) * proj_tol
            ] = 0
        
        print("Everything before this statement has been crosschecked with merger/coordep") 
        # Now run the TZ force constant transformation
        print(os.getcwd())
        zmat_obj2 = Zmat(options)
        zmat_obj2.run(zmat_name="zmat2")
        
        print(options.man_proj)
        options.man_proj = True
        
        
        s_vec = SVectors(
            zmat_obj2, options, zmat_obj2.variable_dictionary_init
        )
        s_vec.run(zmat_obj2.cartesians_init, True, proj=TED_obj.proj)
                
        TED_obj = TED(s_vec.proj, zmat_obj2)
                
        g_mat = GMatrix(zmat_obj2, s_vec, options)
        g_mat.run()
        
        if len(sym_sort) > 1:
            flat_sym_sort = np.array([])
            for i in range(len(sym_sort)):
                flat_sym_sort = np.append(flat_sym_sort,sym_sort[i])
            flat_sym_sort = flat_sym_sort.astype(int)
        
        G = g_mat.G.copy()
        Gtz = G.copy()
        init_bool = False
        if os.path.exists(rootdir + "/fc2.dat"):
            f_read_obj = FcRead("fc2.dat")
        elif os.path.exists(rootdir + "/FCMFINAL2"):
            f_read_obj = FcRead("FCMFINAL2")
        else:
            raise RuntimeError
        
        if not init_bool:
            f_read_obj.run()
            f_conv_obj = FcConv(
                f_read_obj.fc_mat,
                s_vec,
                zmat_obj2,
                "internal",
                False,
                TED_obj,
                options.units,
                False,
            )
            f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_init.FC
        
        
        
        # redundant basis 
        G = np.dot(np.dot(TED_obj.proj.T,G),TED_obj.proj)
        if len(sym_sort) > 1:
            G = G[flat_sym_sort]
            G = G[:,flat_sym_sort]
        print("Giraffe G")
        G[np.abs(G) < 1.0e-10] = 0 
        print(G)
        G = np.dot(np.dot(eig_inv, G), eig_inv.T)
        # G[np.abs(G) < options.tol] = 0
        F = np.dot(np.dot(TED_obj.proj.T,F),TED_obj.proj)
        if len(sym_sort) > 1:
            F = F[flat_sym_sort]
            F = F[:,flat_sym_sort]
        print("Giraffe F")
        F[np.abs(F) < 1.0e-5] = 0 
        print(F)
        F = np.dot(np.dot(inv(eig_inv).T, F), inv(eig_inv))
        # F[np.abs(F) < options.tol] = 0
         
        full_GF = GFMethod(
            G,
            F,
            options.tol,
            options.proj_tol,
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
        
        m = 2 
        var = 0.95 
        
        # def checkted(ted):
            # temps = []
            # for i in range(0,np.shape(ted)[0]):
                # ted_slice = ted[:,i] 
                # temp = copy.copy(ted_slice)
                # for j in range(0,m):
                    # largest = np.argmax(temp)
                    # if temp[largest] < 0.9:
                        # print('not big enough')
                    # print('largest')
                    # print(largest,temp[largest])
                    # temps.append([i,largest])
                    # temp[largest] = 0
                    # print('another')
                    # print(temp)
            # return temps
        # print('is this the ted im looking for?')
        # ted_breakdown = init_GF.ted_breakdown
        # print(ted_breakdown)  
        
        # temps = checkted(ted_breakdown) 
        
        # print('temps')
        # print(temps)
        
        self.reference_freq = full_GF.freq 
        if options.coords == 'Redundants':
            L_B = full_GF.L
        elif options.coords == 'Custom':
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
        
        print("Full Force constant matrix in lower level normal mode basis:")
        print(F)
        if options.coords == 'Redundant':
            self.F_redundant = F
            #self.F_redundant_cma2IDX = n_largest(2, np.abs(copy.copy(self.F_redundant)))
        elif options.coords == 'Custom':
            self.F_custom = F
            #self.F_custom_cma2IDX = n_largest(2, np.abs(copy.copy(self.F_custom)))
        elif options.coords == 'ZMAT' :
            self.F_zmat = F
        else:
            pass
        Fdiag = copy.copy(np.diag(np.diag(F)))
        
        print("Diagonal Force constant matrix in lower level normal mode basis:")
        print(Fdiag)
        diag_GF = GFMethod(
            G,
            Fdiag,
            options.tol,
            options.proj_tol,
            zmat_obj2,
            TED_obj,
            False
        )
        
        diag_GF.run()
        if options.coords == 'Redundant':
            self.Freq_redundant = diag_GF.freq
        elif options.coords == 'Custom':
            self.Freq_custom = diag_GF.freq
        elif options.coords == 'ZMAT' :
            self.Freq_zmat = diag_GF.freq
        else:
            pass
        
        self.Freq_cma2 = diag_GF.freq
        
        # Print Diagonal TED here in projected basis

        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" CMA-0 TED"))
        print("////////////////////////////////////////////")
        TED_obj.run(np.dot(init_GF.L, diag_GF.L), diag_GF.freq, rect_print=False)


        if self.options.n_cma2 > 0:
            if self.options.off_diag:
                print('this is the option')
                print(self.options.off_diag)
                algo = Algorithm(eigs, None, options)
                algo.run()
                print('algo indices')
                print(algo.indices)
                temp = np.zeros((eigs,eigs))  
                print('temp')
                print(temp) 
                for z, extra in enumerate(algo.indices):
                    element = F[extra[0], extra[1]] 
                    temp[extra[0], extra[1]] = element
                    temp[extra[1], extra[0]] = element
                print('temp')
                print(temp)
            else:
                # extras = n_largest(self.options.n_cma2, np.abs(copy.copy(F)))
                #91_ccsd extras = [[4,5],[10,11],[14,16]] 
                #72_ccsd extras = [[4,5]]
                #90_ccsd extras = [[15,17]]
                #2.16_ccsd extras = [[7,9],[8,9]]
                #10_ccsd extras = [[12,15]]
                #82_ccsd extras = [[14,15]]
                #70_ccsd extras = [[10,11],[0,1]]
                #85_ccsd extras = [[1,4],[3,5]]
                extras = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[3,4],[3,5],[3,6],[3,7],[3,8],[4,5],[4,6],[4,7],[4,8],[5,6],[5,7],[5,8],[6,7],[6,8],[7,8]]
                # extras = [[0,1],[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]]
                # print(np.array(extras).shape)
                # extras = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
                #extras = [[17,19]]
                #extras = [[0,2]]
                print('extras')
                print(extras)
                # raise RuntimeError
                temp = copy.copy(Fdiag)
                if len(self.options.other_F_matrix) and os.path.exists(os.getcwd() + "/inter_fc.dat"):
                    print(os.getcwd())
                    f_read_obj_inter = FcRead("inter_fc.dat")
                    f_read_obj_inter.run()
                    F_inter = f_read_obj_inter.fc_mat
                    F_inter = np.dot(np.dot(inv(eig_inv).T, F_inter), inv(eig_inv))
                    print("F_inter:")
                    print(F_inter)
                    print("F_A:")
                    print(F)
                    # raise RuntimeError
                    for z, extra in enumerate(extras):
                        element = F_inter[extra[0], extra[1]] 
                        temp[extra[0], extra[1]] = element
                        temp[extra[1], extra[0]] = element
                else:
                    for z, extra in enumerate(extras):
                        element = F[extra[0], extra[1]] 
                        temp[extra[0], extra[1]] = element
                        temp[extra[1], extra[0]] = element
                print('CMA2 FC matrix')
                print(temp) 
            #if options.coords == 'Redundant':
            #    #F[index] = self.F_redundant[index]     
            #if options.coords == 'Custom':
            #    extras = n_largest(2, np.abs(copy.copy(self.F_custom)))
            #    #F[index] = self.F_custom[index]     
            #elif options.coords == 'ZMAT' :
            #    extras = n_largest(2, np.abs(copy.copy(self.F_zmat)))
            #    #F[index] = self.F_zmat[index]     
            #else:
            #    pass
            print('Time for some off-diags')
            init_GF = GFMethod(
                G,
                temp,
                options.tol,
                options.proj_tol,
                zmat_obj2,
                TED_obj,
                False
            )
            init_GF.run()
            print('CMA2 including ' + str(z + 1) + ' off-diagonal bands/elements for ' + str(options.coords) + ' coordinates')
            print(init_GF.freq) 
        
        def n_largest(n, FC):
            indexes = []
            upper_triang = abs(np.triu(FC,1))
            #print('this is the upper triang')
            #print(upper_triang)
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

