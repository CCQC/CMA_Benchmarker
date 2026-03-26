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
from concordantmodes.sapelo_template import SapeloTemplate
from concordantmodes.s_vectors import SVectors
from concordantmodes.submit import Submit
from concordantmodes.symmetry import Symmetry
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.zmat import Zmat

import copy
from fractions import Fraction


class Merger(object):

    def __init__(self, cmaA_path=None):
        options_kwargs = {
            "queue": "gen4.q,gen6.q,gen5.q",
            "program": "molpro@2010.1.67+mpi",
            "energy_regex": r"\(T\) energy\s+(\-\d+\.\d+)",
            "energy_regex": r"\s*\!CCSD\(T\) total energy\s+(-\d+\.\d+)",
            "cart_insert": 7,
            "calc": False,
            # "calc_init": False,
            "success_regex": r"Variable memory released",
            "cluster": "sapelo",
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
        self.cmaA_path = cmaA_path

    # function that returns diagonal fc matrix + n-largest off-diagonal elements
    def run(
        self,
        opts,
        Proj,
        energy_regex=None,
        success_regex=None,
        cmaA_coord=None,
        sym_sort=None,
        xi_tol=[],
        coord_type_b="internal",
        od_inds=[],
        tiles=[],
        tile_type=[],
        tile_xi={},
    ):

        self.coord_type_b = coord_type_b

        self.Proj = Proj
        # Non-abelian symmetries can be specified with 
        # pairs or trios (or larger) within an array
        if len(sym_sort) > 1:
            flat_sym_sort = np.array([])
            flat_sym_sort_inv = np.array([])
            for i in range(len(sym_sort)):
                if len(np.shape(sym_sort[i])) > 1:
                    flat_sym_sort = np.append(flat_sym_sort, np.array(sym_sort[i]).T.flatten())
                else:
                    flat_sym_sort = np.append(flat_sym_sort, sym_sort[i])
            flat_sym_sort = flat_sym_sort.astype(int)
            for i in range(len(flat_sym_sort)):
                if len(np.where(flat_sym_sort == i)[0]):
                    flat_sym_sort_inv = np.append(
                        flat_sym_sort_inv, np.where(flat_sym_sort == i)[0][0]
                    )
            flat_sym_sort_inv = flat_sym_sort_inv.astype(int)
            # raise RuntimeError

        self.options = opts
        rootdir = os.getcwd()
        zmat_obj = Zmat(self.options)
        zmat_obj.run()

        np.set_printoptions(edgeitems=60, linewidth=1000)

        # Compute the initial s-vectors
        s_vec = SVectors(zmat_obj, self.options)#, zmat_obj.variable_dictionary_b)
        if len(np.shape(self.Proj)) > 2:
            print("this is proj that has been manually sorted by symmetry irrep")
        else:
            print("this is proj, check for this when redundants executed")
            print(self.Proj)

        s_vec.run(
            zmat_obj.cartesians_b,
            True,
            proj=self.Proj,
            second_order=self.options.second_order,
        )
        if self.options.second_order:
            s_vec_b = s_vec
        TED_obj = TED(s_vec.proj, zmat_obj, self.options)
        print("TED PROJ:")
        print(TED_obj.proj)

        g_mat = GMatrix(zmat_obj, s_vec, self.options)
        g_mat.run()

        G = g_mat.G.copy()

        if os.path.exists(rootdir + "/fc.grad"):
            print("FC GRAD EXISTS")
            # raise RuntimeError
            g_read_obj = GrRead("fc.grad")
            g_read_obj.run(zmat_obj.cartesians_b)
            self.options.cart_fc_b = True
            # print(zmat_obj.cartesians_init)

        symm_obj = Symmetry(zmat_obj, self.options, s_vec.proj)
        symm_obj.dummy_obj()
        symm_obj.symtext = None

        init_bool = False
        if os.path.exists(rootdir + "/fc.dat"):
            f_read_obj = FcRead("fc.dat")
        elif os.path.exists(rootdir + "/FCMFINAL"):
            f_read_obj = FcRead("FCMFINAL")
        else:
            init_bool = True
            if cmaA_coord == None:
                print(
                    "You need to specify the cmaA_coord variable for this feature. Check execMerger.run()"
                )
                raise RuntimeError

            os.chdir(os.getcwd() + self.cmaA_path)
            if (
                os.path.exists(os.getcwd() + "/fc_int_" + cmaA_coord + ".dat")
                and not self.options.second_order
            ):
                if os.path.exists(os.getcwd() + "/DispsB"):
                    shutil.rmtree("DispsB")
                f_read_obj = FcRead("fc_int_" + cmaA_coord + ".dat")
                f_read_obj.run()
                fc_b = ForceConstant(
                    None,
                    [],
                    [],
                    0,
                    self.options,
                    [],
                    [],
                )
                fc_b.FC = f_read_obj.fc_mat
                os.chdir("..")
                os.chdir("..")
            elif (
                os.path.exists(os.getcwd() + "/fc_cart.dat")
                and self.options.second_order
            ):
                if os.path.exists(os.getcwd() + "/DispsB"):
                    shutil.rmtree("DispsB")
                f_read_obj = FcRead("fc_cart.dat")
                f_read_obj.run()
                fc_b = ForceConstant(
                    None,
                    [],
                    [],
                    0,
                    self.options,
                    [],
                    [],
                )
                fc_b.FC = f_read_obj.fc_mat
                os.chdir("..")
                os.chdir("..")
            else:
                # First generate displacements in internal coordinates
                eigs_b = np.eye(len(s_vec.proj.T))
                if not self.options.deriv_level_b:
                    indices = np.triu_indices(len(s_vec.proj.T))
                    indices = np.array(indices).T
                    print("symmetric displacements:")
                    if len(sym_sort) > 1:
                        sym_disps = []
                        for i in sym_sort:
                            for j in indices:
                                if len(np.array(i).shape) > 1:
                                    # buff_irrep = np.array(i).flatten()
                                    buff_irrep = np.array([])
                                    # We ought to be able to only utilize the first index of the degenerate group for all displacements.
                                    # Assuming the degenerate NICs are properly aligned.
                                    for k in np.array(i):
                                       buff_irrep = np.append(buff_irrep,k[0])
                                    buff_irrep = buff_irrep.astype(int)
                                else:
                                    buff_irrep = i
                                if j[0] in buff_irrep and j[1] in buff_irrep:
                                    sym_disps.append([j[0], j[1]])
                                    # if j[1] in i:
                                    # sym_disps.append([j[0],j[1]])
                        indices = sym_disps

                else:
                    indices = np.arange(len(eigs_b))
                if self.options.second_order:
                    indices = np.triu_indices(len(zmat_obj.cartesians_b.flatten()))
                    indices = np.array(indices).T
                print(s_vec.B)
                b_disp = TransfDisp(
                    s_vec,
                    zmat_obj,
                    eigs_b,
                    True,
                    TED_obj,
                    self.options,
                    indices,
                    symm_obj=symm_obj,
                    deriv_level=self.options.deriv_level_b,
                    coord_type=self.coord_type_b,
                    cma_level="B"
                )
                b_disp.run()
                # raise RuntimeError
                prog_b = self.options.program_b
                prog_name_b = prog_b.split("@")[0]
                # rootdir = os.getcwd()
                # self.options.calc_init = False
                # print("CALC INIT")
                # print(self.options.calc_init)
                if self.options.calc_b:
                    if os.path.exists(os.getcwd() + "/DispsB"):
                        shutil.rmtree("DispsB")
                    dir_obj_b = DirectoryTree(
                        prog_name_b,
                        zmat_obj,
                        b_disp.disp_cart["ref"],
                        "B",
                        b_disp.p_disp,
                        b_disp.m_disp,
                        self.options,
                        indices,
                        symm_obj,
                        "templateInit.dat",
                        "DispsB",
                        deriv_level=self.options.deriv_level_b,
                    )
                    dir_obj_b.run()
                    # raise RuntimeError
                    disp_list = []
                    for i in os.listdir(os.getcwd()):
                        disp_list.append(i)

                    if self.options.cluster != "sapelo":
                        v_template = VulcanTemplate(
                            self.options, len(disp_list), prog_name_b, prog_b
                        )
                        out = v_template.run()
                        with open("displacements.sh", "w") as file:
                            file.write(out)

                        # Submits an array, then checks if all jobs have finished every
                        # 10 seconds.
                        sub = Submit(disp_list, self.options)
                        sub.run()
                    else:
                        s_template = SapeloTemplate(
                            self.options, len(disp_list), prog_name_b, prog_b
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
                        sub_dir = os.getcwd()
                        print(os.getcwd())
                        sub = Submit(self.options, "B", sub_dir[:-7], prog_name_b, prog_b)
                        sub.run()

                if coord_type_b == "internal":
                    num_deg_free = s_vec.proj.shape[1]
                else:
                    num_deg_free = s_vec.B.shape[1]
                # print(np.dot(s_vec.proj.T,s_vec.B).shape)

                reap_obj_b = Reap(
                    self.options,
                    num_deg_free,
                    indices,
                    symm_obj,
                    "B",
                    deriv_level=self.options.deriv_level_b,
                )
                reap_obj_b.energy_regex = energy_regex
                reap_obj_b.success_regex = success_regex
                if self.options.calc_b:
                    reap_obj_b.run()
                    os.chdir("..")
                else:
                    print(os.getcwd())
                    os.chdir("DispsB")
                    reap_obj_b.run()
                    os.chdir("..")

                # nate
                if not self.options.deriv_level_b:
                    p_array_b = reap_obj_b.p_en_array
                    m_array_b = reap_obj_b.m_en_array
                    ref_en_b = reap_obj_b.ref_en
                else:
                    cart_p_array_b = reap_obj_b.p_grad_array
                    cart_m_array_b = reap_obj_b.m_grad_array
                    p_array_b = np.zeros(np.eye(len(eigs_b)).shape)
                    m_array_b = np.zeros(np.eye(len(eigs_b)).shape)
                    ref_en_b = None
                    # Need to convert this array here from cartesians to internals using projected A-tensor
                    for i in indices:
                        grad_s_vec = SVectors(
                            zmat_obj, self.options#, zmat_obj.variable_dictionary_b
                        )
                        grad_s_vec.run(b_disp.p_disp[i], False)
                        A_proj = np.dot(LA.pinv(grad_s_vec.B), TED_obj.proj)
                        p_array_b[i] = np.dot(cart_p_array_b[i].T, A_proj)
                        grad_s_vec.run(b_disp.m_disp[i], False)
                        A_proj = np.dot(LA.pinv(grad_s_vec.B), TED_obj.proj)
                        m_array_b[i] = np.dot(cart_m_array_b[i].T, A_proj)

                fc_b = ForceConstant(
                    b_disp,
                    p_array_b,
                    m_array_b,
                    ref_en_b,
                    self.options,
                    indices,
                    deriv_level=self.options.deriv_level_b,
                    coord_type_b=coord_type_b,
                    cma_level="B",
                )
                fc_b.run()
                print("Computed Force Constants:")
                print(fc_b.FC.shape)
                print(fc_b.FC)
                # raise RuntimeError
                if self.options.second_order:
                    p_array_grad = np.array([])
                    m_array_grad = np.array([])
                    for i in range(len(p_array_b)):
                        p_array_grad = np.append(p_array_grad, p_array_b[i, i])
                        m_array_grad = np.append(m_array_grad, m_array_b[i, i])
                    grad_b = ForceConstant(
                        b_disp,
                        p_array_grad,
                        m_array_grad,
                        ref_en_b,
                        self.options,
                        indices,
                        deriv_level=1,
                        cma_level="B",
                        coord_type_b=coord_type_b,
                    )
                    grad_b.run()
                    print("Computed Gradient:")
                    print(grad_b.FC)
                f_conv_obj = FcConv(
                    fc_b.FC,
                    s_vec,
                    zmat_obj,
                    "internal",
                    False,
                    TED_obj,
                    self.options,
                )
                f_conv_obj.N = len(fc_b.FC)
                if self.coord_type_b == "internal":
                    f_conv_obj.print_const(fc_name="fc_int_" + cmaA_coord + ".dat")
                    shutil.move(
                        os.getcwd() + "/fc_int_" + cmaA_coord + ".dat",
                        os.getcwd()
                        + "/.."
                        + self.cmaA_path
                        + "/fc_int_"
                        + cmaA_coord
                        + ".dat",
                    )
                elif self.coord_type_b == "cartesian":
                    f_conv_obj.print_const(fc_name="fc_cart.dat")
                    shutil.move(
                        os.getcwd() + "/fc_cart.dat",
                        os.getcwd() + "/.." + self.cmaA_path + "/fc_cart.dat",
                    )
                print("Force Constants saved at:")
                print(self.cmaA_path)
                if self.options.second_order:
                    fc_name = "fc_cart.grad"
                    fc_output = ""
                    g_print = grad_b.FC.copy()
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
                    shutil.move(
                        os.getcwd() + "/fc_cart.grad",
                        os.getcwd() + "/.." + self.cmaA_path + "/fc_cart.grad",
                    )
                    print("Gradient saved at:")
                    print(self.cmaA_path)
                    f_conv_obj.run(grad=grad_b.FC)
                    g_read_obj = GrRead("fc.grad")
                    g_read_obj.cart_grad = grad_b.FC
                    self.options.cart_fc_b = True

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
                self.options,
            )
            if self.options.second_order:
                f_conv_obj.run(grad=g_read_obj.cart_grad)
                # raise RuntimeError
            else:
                f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_b.FC
            if self.options.second_order:
                F = f_conv_obj.F
                # F = np.dot(TED_obj.proj.T, np.dot(F, TED_obj.proj))

        self.options.deriv_level_b = 0

        # if self.options.coords != "ZMAT" and not init_bool:
            # # F = np.dot(TED_obj.proj.T, np.dot(F, TED_obj.proj))
            # # F[np.abs(F) < 1.0e-6] = 0
            # # print("Nat Int force constants:")
            # # print(F)
            # if self.options.second_order:
                # grad_proj = np.dot(TED_obj.proj.T, f_conv_obj.v_q)
        

        if self.options.coords != "ZMAT":
            # g_mat.G = np.dot(TED_obj.proj.T, np.dot(g_mat.G, TED_obj.proj))
            G = np.dot(TED_obj.proj.T, np.dot(G, TED_obj.proj))

        # TED_obj.run(np.eye(TED_obj.proj.shape[1]), np.zeros(TED_obj.proj.shape[1]))

        # raise RuntimeError
        # F_init = F.copy()                
        if len(sym_sort) > 1:
            Fbuff1 = np.array([])
            Gbuff1 = np.array([])
            Fbuff2 = {}
            Gbuff2 = {}
            # print(sym_sort)
            # print(flat_sym_sort)
            # print(flat_sym_sort)
            # print(flat_sym_sort_inv)
            for i in range(len(sym_sort)):
                Fbuff1 = F.copy()
                # sym_sort_buff = sym_sort[i].copy()
                # print(sym_sort_buff)
                # rase RuntimeError
                
                # Separate logic here to average the FC blocks of degenerate blocks
                if len(np.shape(sym_sort[i])) > 1:
                    # sym_sort_buff = np.array(sym_sort_buff).flatten()
                    # print(sym_sort_buff)
                    sym_sort_T = np.array(sym_sort[i]).T
                    # print(sym_sort_T)
                    Fbuff_degen = np.zeros(len(sym_sort_T), dtype=object)
                    for j in range(len(sym_sort_T)):
                        # print(sym_sort_T[j])
                        Fbuff1 = Fbuff1[sym_sort_T[j]]
                        Fbuff1 = Fbuff1[:,sym_sort_T[j]]
                        Fbuff_degen[j] = Fbuff1
                        Fbuff1 = F.copy()
                    # F_ave = Fbuff_degen[0]
                    # F_ave = np.sum(np.abs(Fbuff_degen),axis=0)/len(sym_sort_T)

                    F_ave = np.sum(Fbuff_degen,axis=0)/len(sym_sort_T)
                    # print(F_ave.shape)
                    # print(F_ave)
                    for j in range(len(sym_sort_T)):
                        Fbuff_degen[j] = F_ave
                    Fbuff1 = Fbuff_degen[0].copy()
                    # Fbuff = Fbuff1.copy()
                    for j in range(len(sym_sort_T)-1):
                        Fbuff1 = np.block(
                            [
                                [Fbuff1, np.zeros((len(Fbuff1), len(Fbuff_degen[j + 1])))],
                                [
                                    np.zeros((len(Fbuff_degen[j + 1]), len(Fbuff1))),
                                    Fbuff_degen[j + 1],
                                ],
                            ]
                        )
                    
                    print("F:")
                    print(Fbuff1.shape)
                    print(Fbuff1)
                    Fbuff2[str(i)] = Fbuff1.copy()
                    
                    Gbuff1 = G.copy()
                    Gbuff_degen = np.zeros(len(sym_sort_T), dtype=object)
                    for j in range(len(sym_sort_T)):
                        Gbuff1 = Gbuff1[sym_sort_T[j]]
                        Gbuff1 = Gbuff1[:,sym_sort_T[j]]
                        Gbuff_degen[j] = Gbuff1
                        Gbuff1 = G.copy()
                    # G_ave = Gbuff_degen[0]
                    G_ave = np.sum(Gbuff_degen,axis=0)/len(sym_sort_T)
                    # G_ave = np.sum(np.abs(Gbuff_degen),axis=0)/len(sym_sort_T)
                    for j in range(len(sym_sort_T)):
                        Gbuff_degen[j] = G_ave
                    Gbuff1 = Gbuff_degen[0].copy()
                    # Gbuff = Gbuff1.copy()
                    for j in range(len(sym_sort_T)-1):
                        Gbuff1 = np.block(
                            [
                                [Gbuff1, np.zeros((len(Gbuff1), len(Gbuff_degen[j + 1])))],
                                [
                                    np.zeros((len(Gbuff_degen[j + 1]), len(Gbuff1))),
                                    Gbuff_degen[j + 1],
                                ],
                            ]
                        )
                    
                    print("G:")
                    print(Gbuff1.shape)
                    print(Gbuff1)
                    Gbuff2[str(i)] = Gbuff1.copy()

                else:
                    Fbuff1 = F.copy()
                    Fbuff1 = Fbuff1[sym_sort[i]]
                    Fbuff1 = np.array(Fbuff1[:, sym_sort[i]])
                    # print(i)
                    # print(Fbuff1.shape)
                    # print(Fbuff1)
                    Fbuff2[str(i)] = Fbuff1.copy()
                    Gbuff1 = G.copy()
                    Gbuff1 = Gbuff1[sym_sort[i]]
                    Gbuff1 = np.array(Gbuff1[:, sym_sort[i]])
                    Gbuff2[str(i)] = Gbuff1.copy()
                
                


                # Fbuff1 = F.copy()
                # Fbuff1 = Fbuff1[sym_sort_buff]
                # Fbuff1 = np.array(Fbuff1[:, sym_sort_buff])
                # Fbuff2[str(i)] = Fbuff1.copy()
                # Gbuff1 = G.copy()
                # Gbuff1 = Gbuff1[sym_sort_buff]
                # Gbuff1 = np.array(Gbuff1[:, sym_sort_buff])
                # Gbuff2[str(i)] = Gbuff1.copy()
            
            Fbuff3 = Fbuff2[str(0)].copy()
            Gbuff3 = Gbuff2[str(0)].copy()
            for i in range(len(sym_sort) - 1):
                Fbuff3 = np.block(
                    [
                        [Fbuff3, np.zeros((len(Fbuff3), len(Fbuff2[str(i + 1)])))],
                        [
                            np.zeros((len(Fbuff2[str(i + 1)]), len(Fbuff3))),
                            Fbuff2[str(i + 1)],
                        ],
                    ]
                )
                Gbuff3 = np.block(
                    [
                        [Gbuff3, np.zeros((len(Gbuff3), len(Gbuff2[str(i + 1)])))],
                        [
                            np.zeros((len(Gbuff2[str(i + 1)]), len(Gbuff3))),
                            Gbuff2[str(i + 1)],
                        ],
                    ]
                )
                
            np.set_printoptions(precision=4, linewidth=350)
            print(sym_sort)
            print("Full F:")
            print(Fbuff3.shape)
            print(Fbuff3)
            # F = Fbuff3.copy()
            F = Fbuff3[flat_sym_sort_inv]
            F = F[:, flat_sym_sort_inv]
            np.set_printoptions(precision=8, linewidth=500)
            print(sym_sort)
            print("Full G:")
            print(Gbuff3.shape)
            print(Gbuff3)
            # g_mat.G = Gbuff3.copy()
            # G = Gbuff3.copy()
            G = Gbuff3[flat_sym_sort_inv]
            G = G[:, flat_sym_sort_inv]
            # g_mat.G = Gbuff3[flat_sym_sort_inv]
            # g_mat.G = g_mat.G[:, flat_sym_sort_inv]

        # raise RuntimeError


        if False:
        # if len(sym_sort) > 1:
            print(F.shape)
            print(len(flat_sym_sort)) 
            print(flat_sym_sort) 
            F_sym = F[flat_sym_sort].copy()
            F_sym = F_sym[:, flat_sym_sort]
            print("Sym Force Constants:")
            print(F_sym)
            # F_sym = F_sym[flat_sym_sort.argsort()].copy()
            # F_sym = F_sym[:,flat_sym_sort.argsort()]

            g_sym = g_mat.G[flat_sym_sort].copy()
            g_sym = g_sym[:, flat_sym_sort]
            g_sym[np.abs(g_sym) < 1e-9] = 0
            print("Sym G-Matrix:")
            print(sym_sort)
            print(g_sym)

        print("Initial Force Constants:")
        print(F.shape)
        # print(F[26,27])
        print(F)

        print("Initial G-Matrix:")
        # g_mat.G[np.abs(g_mat.G) < 1e-9] = 0
        print(G.shape)
        # print(g_mat.G)
        # print(G[26,27])
        print(G)

        print("Initial Frequencies:")
        # init_GF = GFMethod(g_mat.G.copy(), F.copy(), zmat_obj, TED_obj, self.options)
        print(self.options.molsym_symmetry)
        b_GF = GFMethod(G.copy(), F.copy(), zmat_obj, TED_obj, self.options)
        b_GF.run()

        # raise RuntimeError

        # print("L")
        # degen = False
        # for i in range(len(sym_sort)):
            # if len(np.shape(sym_sort[i])) > 1:
                # degen = True
        # if len(sym_sort):
        # # if degen:
            # init_GF.L = init_GF.L[flat_sym_sort_inv]
            # F = init_GF.F[flat_sym_sort_inv]
            # F = F[:,flat_sym_sort_inv]
            # G = init_GF.G[flat_sym_sort_inv]
            # G = G[:,flat_sym_sort_inv]
            # init_GF.ted.TED = init_GF.ted.TED[flat_sym_sort_inv]
        # print(init_GF.L)
        # print(init_GF.L[flat_sym_sort])
        # print("L_p")
        # print(init_GF.L_p)
        # print(init_GF.L_p)
        # print(init_GF.L_p[flat_sym_sort])
        # print("F_O")
        # print(init_GF.F_O)
        # F_O = init_GF.F_O[flat_sym_sort]
        # print(F_O[:,flat_sym_sort])
        # print("TED for sym purposes: ")
        b_GF.ted.TED[np.abs(b_GF.ted.TED) < 1e-5] = 0
        print(b_GF.ted.TED)
        # print(sym_sort)
        # print(flat_sym_sort)
        # print(flat_sym_sort_inv)
        # print(flat_sym_sort[flat_sym_sort_inv])
        # print(flat_sym_sort_inv[flat_sym_sort])
        # raise RuntimeError

        self.ref_b = b_GF.freq
        if len(sym_sort):
            self.irreps_b, flat_sym_freqs = self.mode_symmetry_sort(
                b_GF.ted.TED, sym_sort, self.ref_b
            )
            self.ref_b = np.array(flat_sym_freqs)
        if len(tiles):
            self.tiles_b, sorted_freqs = self.mode_symmetry_sort(
                b_GF.ted.TED, tiles, self.ref_b, percent_tol=80.0
            )
            # print(tiles)
            # print(self.tiles_init)
            # raise RuntimeError

        # Now for the TED check.
        G = np.dot(np.dot(LA.inv(b_GF.L), G), LA.inv(b_GF.L).T)
        G[np.abs(G) < self.options.tol] = 0
        print(G)
        # print(self.irreps_init)
        # raise RuntimeError
        F = np.dot(np.dot(b_GF.L.T, F), b_GF.L)
        F[np.abs(F) < self.options.tol] = 0
        # if self.options.second_order:
        # grad_n = np.dot(init_GF.L.T, grad_proj)
        # print("Normal Mode Gradients:")
        # for i in range(len(grad_n)):
        # print(str(i+1) + ": " + str(grad_n[i]))

        # Try to average the force constants and G matrix here.

        print("TED Frequencies:")
        TED_GF = GFMethod(G, F, zmat_obj, TED_obj, self.options)
        TED_GF.run()

        # raise RuntimeError

        proj_tol = 1.0e-3
        eig_inv = inv(b_GF.L)  # (Normal modes (Q) x Sym internals (S) )
        for i in range(len(eig_inv)):
            eig_inv[i] = eig_inv[i] / LA.norm(eig_inv[i])
            eig_inv[i][np.abs(eig_inv[i]) < np.max(np.abs(eig_inv[i])) * proj_tol] = 0

        # Now run the TZ force constant transformation
        zmat_obj2 = Zmat(self.options)
        zmat_obj2.run(zmat_name="zmat2")

        self.options.man_proj = True

        s_vec = SVectors(zmat_obj2, self.options)#, zmat_obj2.variable_dictionary_b)
        s_vec.run(zmat_obj2.cartesians_b, True, proj=TED_obj.proj)

        TED_obj = TED(s_vec.proj, zmat_obj2, self.options)

        g_mat = GMatrix(zmat_obj2, s_vec, self.options)
        g_mat.run()

        G = g_mat.G.copy()
        # Gtz = G.copy()
        init_bool = False
        if os.path.exists(rootdir + "/fc2.dat"):
            f_read_obj = FcRead("fc2.dat")
        elif os.path.exists(rootdir + "/FCMFINAL2"):
            f_read_obj = FcRead("FCMFINAL2")
        else:
            raise RuntimeError

        self.options.second_order = False

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
                self.options,
            )
            f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_b.FC

        # redundant basis
        G = np.dot(np.dot(TED_obj.proj.T, G), TED_obj.proj)
        # Conversion to aJ/Ang
        F_aJ = F.copy()
        F_aJ *= 4.3597447222071
        F_aJ /= 0.529177210903
        # F = np.dot(np.dot(TED_obj.proj.T, F), TED_obj.proj)
        # F_aJ = np.dot(np.dot(TED_obj.proj.T, F_aJ), TED_obj.proj)
        # if len(sym_sort) > 1:
        # F = F[flat_sym_sort]
        # F = F[:,flat_sym_sort]
        # F_aJ = F_aJ[flat_sym_sort]
        # F_aJ = F_aJ[:,flat_sym_sort]
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
        # G[np.abs(G) < 1.0e-9] = 0
        # print(G)
        # if len(sym_sort) > 1:
        # print(g_sym)
        
        
        # if len(sym_sort) > 1:
            # Fbuff1 = np.array([])
            # Fbuff2 = {}
            # Gbuff1 = np.array([])
            # Gbuff2 = {}
            # for i in range(len(sym_sort)):
                # Fbuff1 = F.copy()
                # if len(np.array(sym_sort[i]).shape) > 1:
                    # buff_irrep = np.array(sym_sort[i]).flatten()
                # else:
                    # buff_irrep = sym_sort[i]
                # Fbuff1 = Fbuff1[buff_irrep]
                # Fbuff1 = np.array([Fbuff1[:, buff_irrep]])
                # Fbuff2[str(i)] = Fbuff1.copy()
                # Gbuff1 = G.copy()
                # Gbuff1 = Gbuff1[buff_irrep]
                # Gbuff1 = np.array([Gbuff1[:, buff_irrep]])
                # Gbuff2[str(i)] = Gbuff1.copy()
            # Fbuff3 = Fbuff2[str(0)][0].copy()
            # Gbuff3 = Gbuff2[str(0)][0].copy()
            # for i in range(len(sym_sort) - 1):
                # Fbuff3 = np.block(
                    # [
                        # [Fbuff3, np.zeros((len(Fbuff3), len(Fbuff2[str(i + 1)][0])))],
                        # [
                            # np.zeros((len(Fbuff2[str(i + 1)][0]), len(Fbuff3))),
                            # Fbuff2[str(i + 1)][0],
                        # ],
                    # ]
                # )
                # Gbuff3 = np.block(
                    # [
                        # [Gbuff3, np.zeros((len(Gbuff3), len(Gbuff2[str(i + 1)][0])))],
                        # [
                            # np.zeros((len(Gbuff2[str(i + 1)][0]), len(Gbuff3))),
                            # Gbuff2[str(i + 1)][0],
                        # ],
                    # ]
                # )
            # F = Fbuff3[flat_sym_sort_inv]
            # F = F[:, flat_sym_sort_inv]
            # G = Gbuff3[flat_sym_sort_inv]
            # G = G[:, flat_sym_sort_inv]

        # Checking this logic, can bring back later when fixed.
        # if False:
        if len(sym_sort) > 1:
            Fbuff1 = np.array([])
            Gbuff1 = np.array([])
            Fbuff2 = {}
            Gbuff2 = {}
            for i in range(len(sym_sort)):
                Fbuff1 = F.copy()
                # Separate logic here to average the FC blocks of degenerate blocks
                if len(np.shape(sym_sort[i])) > 1:
                    sym_sort_T = np.array(sym_sort[i]).T
                    Fbuff_degen = np.zeros(len(sym_sort_T), dtype=object)
                    for j in range(len(sym_sort_T)):
                        Fbuff1 = Fbuff1[sym_sort_T[j]]
                        Fbuff1 = Fbuff1[:,sym_sort_T[j]]
                        Fbuff_degen[j] = Fbuff1
                        Fbuff1 = F.copy()
                    
                    F_ave = np.sum(Fbuff_degen,axis=0)/len(sym_sort_T)
                    
                    for j in range(len(sym_sort_T)):
                        Fbuff_degen[j] = F_ave
                    
                    Fbuff1 = Fbuff_degen[0].copy()
                    # Fbuff = Fbuff1.copy()
                    for j in range(len(sym_sort_T)-1):
                        Fbuff1 = np.block(
                            [
                                [Fbuff1, np.zeros((len(Fbuff1), len(Fbuff_degen[j + 1])))],
                                [
                                    np.zeros((len(Fbuff_degen[j + 1]), len(Fbuff1))),
                                    Fbuff_degen[j + 1],
                                ],
                            ]
                        )
                    
                    Fbuff2[str(i)] = Fbuff1.copy()
                    
                    Gbuff1 = G.copy()
                    Gbuff_degen = np.zeros(len(sym_sort_T), dtype=object)
                    
                    for j in range(len(sym_sort_T)):
                        Gbuff1 = Gbuff1[sym_sort_T[j]]
                        Gbuff1 = Gbuff1[:,sym_sort_T[j]]
                        Gbuff_degen[j] = Gbuff1
                        Gbuff1 = G.copy()
                    
                    G_ave = np.sum(Gbuff_degen,axis=0)/len(sym_sort_T)
                    
                    for j in range(len(sym_sort_T)):
                        Gbuff_degen[j] = G_ave
                    
                    Gbuff1 = Gbuff_degen[0].copy()
                    Gbuff = Gbuff1.copy()
                    
                    for j in range(len(sym_sort_T)-1):
                        Gbuff1 = np.block(
                            [
                                [Gbuff1, np.zeros((len(Gbuff1), len(Gbuff_degen[j + 1])))],
                                [
                                    np.zeros((len(Gbuff_degen[j + 1]), len(Gbuff1))),
                                    Gbuff_degen[j + 1],
                                ],
                            ]
                        )
                    Gbuff2[str(i)] = Gbuff1.copy()

                else:
                    Fbuff1 = F.copy()
                    Fbuff1 = Fbuff1[sym_sort[i]]
                    Fbuff1 = np.array(Fbuff1[:, sym_sort[i]])
                    Fbuff2[str(i)] = Fbuff1.copy()
                    Gbuff1 = G.copy()
                    Gbuff1 = Gbuff1[sym_sort[i]]
                    Gbuff1 = np.array(Gbuff1[:, sym_sort[i]])
                    Gbuff2[str(i)] = Gbuff1.copy()
                    print("Force Const and G sub blocks:")
                    print(i)
                    print(Fbuff1)
                    print(Gbuff1)
            
            Fbuff3 = Fbuff2[str(0)].copy()
            Gbuff3 = Gbuff2[str(0)].copy()
            for i in range(len(sym_sort) - 1):
                Fbuff3 = np.block(
                    [
                        [Fbuff3, np.zeros((len(Fbuff3), len(Fbuff2[str(i + 1)])))],
                        [
                            np.zeros((len(Fbuff2[str(i + 1)]), len(Fbuff3))),
                            Fbuff2[str(i + 1)],
                        ],
                    ]
                )
                Gbuff3 = np.block(
                    [
                        [Gbuff3, np.zeros((len(Gbuff3), len(Gbuff2[str(i + 1)])))],
                        [
                            np.zeros((len(Gbuff2[str(i + 1)]), len(Gbuff3))),
                            Gbuff2[str(i + 1)],
                        ],
                    ]
                )
            print("Sym G")
            print(Gbuff3)
            print("Sym F")
            print(Fbuff3)
            # raise RuntimeError
            # F = Fbuff3.copy()
            F = Fbuff3[flat_sym_sort_inv]
            F = F[:, flat_sym_sort_inv]
            # G = Gbuff3.copy()
            G = Gbuff3[flat_sym_sort_inv]
            G = G[:, flat_sym_sort_inv]
        
        G = np.dot(np.dot(eig_inv, G), eig_inv.T)
        # G[np.abs(G) < 1.0e-7] = 0
        # print(G[34])
        F = np.dot(np.dot(inv(eig_inv).T, F), inv(eig_inv))
        # F[np.abs(F) < 1.0e-5] = 0
        print("Normal Mode G")
        print(G)
        print("Normal Mode F")
        print(F)
        # F[np.abs(F) < self.options.tol] = 0

        # print('Testing G:')
        # print(G)
        # print('Testing F:')
        # print(F)
        full_GF = GFMethod(G, F, zmat_obj2, TED_obj, self.options)
        full_GF.run()
        # print("Giraffe F_O")
        # print(full_GF.F_O)
        # print("Giraffe L_p")
        # print(full_GF.L_p)
        self.ted = full_GF.ted.TED  # TED matrix

        # if len(sym_sort):
            # full_GF.L = full_GF.L[flat_sym_sort_inv]
            # full_GF.ted.TED = full_GF.ted.TED[flat_sym_sort_inv]
            # F = full_GF.F[flat_sym_sort_inv]
            # F = F[:,flat_sym_sort_inv]
            # G = full_GF.G[flat_sym_sort_inv]
            # G = G[:,flat_sym_sort_inv]
        
        # G = np.dot(np.dot(eig_inv, G), eig_inv.T)
        # G[np.abs(G) < 1.0e-7] = 0
        # F = np.dot(np.dot(inv(eig_inv).T, F), inv(eig_inv))
        # F[np.abs(F) < 1.0e-5] = 0

        # Print Full TED here in projected basis

        # print("////////////////////////////////////////////")
        # print("//{:^40s}//".format(" Normal Mode TED"))
        # print("////////////////////////////////////////////")
        # TED_obj.run(np.dot(init_GF.L.T, full_GF.L), full_GF.freq, rect_print=False)
        
        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" Full Hessian TED"))
        print("////////////////////////////////////////////")
        # TED_obj.run(full_GF.L, full_GF.freq, rect_print=False)
        # TED_obj.run(np.dot(init_GF.L.T, full_GF.L), full_GF.freq, rect_print=False)
        TED_obj.run(np.dot(b_GF.L, full_GF.L), full_GF.freq, rect_print=False)
        # raise RuntimeError

        self.reference_freq = full_GF.freq
        if len(sym_sort):
            self.irreps_ref, flat_sym_freqs = self.mode_symmetry_sort(
                TED_obj.TED, sym_sort, self.reference_freq
            )
            self.reference_freq = np.array(flat_sym_freqs)
        if len(tiles):
            self.tiles_ref, sorted_freqs = self.mode_symmetry_sort(
                TED_obj.TED, tiles, self.reference_freq, percent_tol=80.0
            )
            # print(tiles)
            # print(self.tiles_ref)
            # raise RuntimeError

        m = 2
        var = 0.95

        self.reference_TED = TED_obj.TED
        ref_TED = self.reference_TED

        # raise RuntimeError
        if self.options.coords == "Redundants":
            L_B = full_GF.L
        elif self.options.coords == "Custom":
            L_A = full_GF.L

        def n_largest(n, FC):
            indexes = []
            upper_triang = abs(np.triu(FC, n))
            for i in range(0, n):
                fc_cma2 = np.where(upper_triang == upper_triang.max())
                index = [fc_cma2[0][0], fc_cma2[1][0]]
                indexes.append(index)
                upper_triang[index[0], index[1]] = 0
            print(indexes)
            return indexes

        np.set_printoptions(edgeitems=60, linewidth=10000)
        print("Full Force constant matrix in lower level normal mode basis:")
        print(F)
        if self.options.coords == "Redundant":
            self.F_redundant = F
        elif self.options.coords == "Custom":
            self.F_custom = F
        elif self.options.coords == "ZMAT":
            self.F_zmat = F
        else:
            pass
        Fdiag = copy.copy(np.diag(np.diag(F)))

        print("Diagonal Force constant matrix in lower level normal mode basis:")
        print(Fdiag)
        diag_GF = GFMethod(G, Fdiag, zmat_obj2, TED_obj, self.options)

        diag_GF.run()

        self.Freq_CMA0 = diag_GF.freq
        # raise RuntimeError
        diag_TED = diag_GF.ted.TED.copy()
        
        self.denom = len(self.Freq_CMA0) * 1.0
        freq_diff = self.Freq_CMA0 - full_GF.freq
        self.outliers = len(freq_diff[np.abs(freq_diff) > 2.5])
        # print(freq_diff[np.abs(freq_diff) > 2.5])
        # print(self.Freq_CMA0)
        # print(full_GF.freq)
        # print(freq_diff)
        # print(len(freq_diff[np.abs(freq_diff) > 2.5]))
        # raise RuntimeError

        if self.options.coords == 'Redundant':
            self.Freq_redundant = diag_GF.freq
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
        TED_obj.run(np.dot(b_GF.L, diag_GF.L), diag_GF.freq, rect_print=False)

        if len(sym_sort):
            self.irreps_CMA0, flat_sym_freqs = self.mode_symmetry_sort(
                TED_obj.TED, sym_sort, self.Freq_CMA0
            )
            self.Freq_CMA0 = np.array(flat_sym_freqs)
        if len(tiles):
            self.tiles_CMA0, sorted_freqs = self.mode_symmetry_sort(
                TED_obj.TED, tiles, self.Freq_CMA0, percent_tol=80.0
            )
            # print(tiles)
            # print(self.tiles_CMA0)

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
                print("Time for some off-diags")
                cmaA_GF = GFMethod(
                    G,
                    temp,
                    zmat_obj2,
                    TED_obj,
                    self.options,
                )
                cmaA_GF.run()
                cmaA_Freq = cmaA_GF.freq.copy()

                print("////////////////////////////////////////////")
                print("//{:^40s}//".format(" CMA-1 TED"))
                print("////////////////////////////////////////////")
                TED_obj.run(
                    np.dot(b_GF.L, cmaA_GF.L), cmaA_GF.freq, rect_print=False
                )

                if len(sym_sort):
                    self.irreps_CMA1, flat_sym_freqs = self.mode_symmetry_sort(
                        TED_obj.TED, sym_sort, cmaA_Freq
                    )
                    cmaA_Freq = np.array(flat_sym_freqs)

                # self.RMSD = np.append(self.RMSD,cmaA_rmsd)
                self.Freq_cmaA = cmaA_Freq

            elif self.options.off_diag == 2:
                self.Freq_cma2 = np.array([])
                self.eta_num = np.array([])
                self.eta_denom = np.array([])
                self.total_off_diags = np.array([])

                for xi_tol_i in xi_tol:
                    print(xi_tol_i)
                    print("Time for some off-diags")
                    # if len(self.options.other_F_matrix) and os.path.exists(os.getcwd() + "/inter_fc.dat"):
                    if len(self.options.other_F_matrix):
                        if (
                            os.path.exists(os.getcwd() + "/inter_fc.dat")
                            and self.coord_type_b == "internal"
                        ):
                            f_read_obj_inter = FcRead("inter_fc.dat")
                            f_read_obj_inter.run()
                            F_inter = f_read_obj_inter.fc_mat
                        elif (
                            os.path.exists(os.getcwd() + "/inter_fc_cart.dat")
                            and self.coord_type_b == "cartesian"
                        ):
                            f_read_obj_inter = FcRead("inter_fc_cart.dat")
                            g_read_obj_inter = GrRead("inter_fc_cart.grad")
                            g_read_obj_inter.run(zmat_obj2.cartesians_b)
                            f_read_obj_inter.run()
                            f_conv = FcConv(
                                f_read_obj_inter.fc_mat,
                                s_vec_b,
                                zmat_obj2,
                                "internal",
                                False,
                                TED_obj,
                                self.options,
                            )
                            f_conv.run(grad=g_read_obj_inter.cart_grad)
                            F_inter = f_conv.F
                            # F_inter = np.dot(
                                # TED_obj.proj.T, np.dot(F_inter, TED_obj.proj)
                            # )

                        F_inter = np.dot(np.dot(inv(eig_inv).T, F_inter), inv(eig_inv))
                        print("F_inter:")
                        print(F_inter)
                        print("F_A:")
                        print(F)
                        xi = F_inter * 0
                        od_inds = []
                        if len(sym_sort) > 1:
                            self.total_off_diags_buff = 0
                            print(self.irreps_b)
                            if len(tiles):
                                print(self.tiles_b)
                                print(self.tiles_CMA0)
                            for irrep in self.irreps_b:
                                if len(irrep) > 1:
                                    for i in range(len(irrep)):
                                        for j in range(i):
                                            if i != j:
                                                a = irrep[i]
                                                b = irrep[j]
                                                if len(tiles):
                                                    for c1 in range(
                                                        len(self.tiles_b)
                                                    ):
                                                        if a in self.tiles_b[c1]:
                                                            break
                                                    for c2 in range(
                                                        len(self.tiles_b)
                                                    ):
                                                        if b in self.tiles_b[c2]:
                                                            break
                                                    # Extremely rudimentary sieve for these interactions.
                                                    # At some point we will need some sort of adjacency matrix for our monomers
                                                    # to ensure that we take advantage of locality in our mixing.
                                                    # raise RuntimeError
                                                    if (
                                                        tile_type[c1] == "i"
                                                        or tile_type[c2] == "i"
                                                    ):
                                                        if c1 == c2:
                                                            # Use xi for same same intermol modes
                                                            xi_tol_i = tile_xi["ii"]
                                                        elif (
                                                            tile_type[c1] == "i"
                                                            and tile_type[c2] == "i"
                                                        ):
                                                            # Use xi for different intermol modes
                                                            xi_tol_i = tile_xi["i1i2"]
                                                        else:
                                                            # Use xi for intra-intermol modes
                                                            xi_tol_i = tile_xi["mi"]
                                                    elif (
                                                        tile_type[c1] == "m1"
                                                        or tile_type[c1] == "m2"
                                                    ):
                                                        if c1 == c2:
                                                            # Use xi value for intramolecular interactions
                                                            if tile_type[c1] == "m1":
                                                                xi_tol_i = tile_xi[
                                                                    "m1m1"
                                                                ]
                                                            elif tile_type[c1] == "m2":
                                                                xi_tol_i = tile_xi[
                                                                    "m2m2"
                                                                ]
                                                        else:
                                                            # Use very large xi value to dampen mixing of intramolecular modes between
                                                            # monomer units.
                                                            xi_tol_i = tile_xi["m1m2"]
                                                    else:
                                                        print(
                                                            "Tile type must be m or i, check what you put in the tile_type array"
                                                        )
                                                        raise RuntimeError
                                                buff = np.abs(F_inter[a, b])
                                                xi[a, b] = buff / np.sqrt(
                                                    np.abs(F_inter[a, a])
                                                    * np.abs(F_inter[b, b])
                                                )
                                                if xi[a, b] > xi_tol_i:
                                                    od_inds.append([a, b])
                                    self.total_off_diags_buff += (
                                        len(irrep) ** 2 - len(irrep)
                                    ) / 2
                        else:
                            # print(tiles)
                            self.total_off_diags_buff = (len(xi) ** 2 - len(xi)) / 2
                            for i in range(len(xi)):
                                for j in range(i + 1):
                                    if i != j:
                                        if len(tiles):
                                            # print(tile_type)
                                            # print(tile_xi)
                                            # print(i,j)
                                            for c1 in range(len(self.tiles_b)):
                                                if i in self.tiles_b[c1]:
                                                    break
                                            for c2 in range(len(self.tiles_b)):
                                                if j in self.tiles_b[c2]:
                                                    break
                                            # print(c1)
                                            # print(c2)
                                            # raise RuntimeError
                                            # Extremely rudimentary sieve for these interactions.
                                            # At some point we will need some sort of adjacency matrix for our monomers
                                            # to ensure that we take advantage of locality in our mixing.
                                            # raise RuntimeError
                                            if (
                                                tile_type[c1] == "i"
                                                or tile_type[c2] == "i"
                                            ):
                                                if c1 == c2:
                                                    # Use xi for same same intermol modes
                                                    xi_tol_i = tile_xi["ii"]
                                                elif (
                                                    tile_type[c1] == "i"
                                                    and tile_type[c2] == "i"
                                                ):
                                                    # Use xi for different intermol modes
                                                    xi_tol_i = tile_xi["i1i2"]
                                                else:
                                                    # Use xi for intra-intermol modes
                                                    xi_tol_i = tile_xi["mi"]
                                            elif (
                                                tile_type[c1] == "m1"
                                                or tile_type[c1] == "m2"
                                            ):
                                                if c1 == c2:
                                                    # Use xi value for intramolecular interactions
                                                    if tile_type[c1] == "m1":
                                                        xi_tol_i = tile_xi["m1m1"]
                                                    elif tile_type[c1] == "m2":
                                                        xi_tol_i = tile_xi["m2m2"]
                                                else:
                                                    # Use very large xi value to dampen mixing of intramolecular modes between
                                                    # monomer units.
                                                    xi_tol_i = tile_xi["m1m2"]
                                            else:
                                                print(
                                                    "Tile type must be m1, m2, or i, check what you put in the tile_type array"
                                                )
                                                print(tile_type[c1])
                                                print(tile_type[c2])
                                                raise RuntimeError
                                        buff = np.abs(F_inter[i, j])
                                        xi[i, j] = buff / np.sqrt(
                                            np.abs(F_inter[i, i])
                                            * np.abs(F_inter[j, j])
                                        )
                                        if xi[i, j] > xi_tol_i:
                                            od_inds.append([i, j])

                        # raise RuntimeError
                        print("CMA2 off-diagonal elements:")
                        print(od_inds)
                        print(len(od_inds))
                        print(self.total_off_diags_buff)
                        print("% ODs")
                        print(len(od_inds) / self.total_off_diags_buff * 100)
                        print("% eta")
                        print(len(od_inds) * 1.0 / len(self.Freq_CMA0) * 100.0)
                        self.cma_off_diags = len(od_inds)
                        # self.off_diags = np.append(self.off_diags,len(od_inds))
                        self.total_off_diags = np.append(
                            self.total_off_diags, self.total_off_diags_buff
                        )
                        # self.perc_off_diags = (self.cma_off_diags / self.total_off_diags) * 100
                        self.eta_num = np.append(self.eta_num, self.cma_off_diags * 1.0)
                        self.eta_denom = np.append(
                            self.eta_denom, len(self.Freq_CMA0) * 1.0
                        )
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
                        zmat_obj2,
                        TED_obj,
                        self.options,
                    )
                    cma2_GF.run()
                    cma2_Freq = cma2_GF.freq.copy()

                    print("////////////////////////////////////////////")
                    print("//{:^40s}//".format(" CMA-2 TED"))
                    print("////////////////////////////////////////////")
                    TED_obj.run(
                        np.dot(b_GF.L, cma2_GF.L), cma2_GF.freq, rect_print=False
                    )

                    if len(sym_sort):
                        self.irreps_CMA2, flat_sym_freqs = self.mode_symmetry_sort(
                            TED_obj.TED, sym_sort, cma2_Freq
                        )
                        cma2_Freq = np.array(flat_sym_freqs)

                    # self.RMSD = np.append(self.RMSD,cma2_rmsd)
                    self.Freq_cma2 = np.append(self.Freq_cma2, cma2_Freq, axis=0)
                self.Freq_cma2 = np.reshape(self.Freq_cma2, (len(xi_tol), -1))
            elif self.options.off_diag == 3:
                self.Freq_cma2 = np.array([])
                self.eta_num = np.array([])
                self.eta_denom = np.array([])
                self.total_off_diags = np.array([])
                if len(self.options.other_F_matrix):
                    if (
                        os.path.exists(os.getcwd() + "/inter_fc.dat")
                        and self.coord_type_b == "internal"
                    ):
                        f_read_obj_inter = FcRead("inter_fc.dat")
                        f_read_obj_inter.run()
                        F_inter = f_read_obj_inter.fc_mat
                    elif (
                        os.path.exists(os.getcwd() + "/inter_fc_cart.dat")
                        and self.coord_type_b == "cartesian"
                    ):
                        f_read_obj_inter = FcRead("inter_fc_cart.dat")
                        g_read_obj_inter = GrRead("inter_fc_cart.grad")
                        g_read_obj_inter.run(zmat_obj2.cartesians_b)
                        f_read_obj_inter.run()
                        f_conv = FcConv(
                            f_read_obj_inter.fc_mat,
                            s_vec_b,
                            zmat_obj2,
                            "internal",
                            False,
                            TED_obj,
                            self.options,
                        )
                        f_conv.run(grad=g_read_obj_inter.cart_grad)
                        F_inter = f_conv.F
                        # F_inter = np.dot(
                            # TED_obj.proj.T, np.dot(F_inter, TED_obj.proj)
                        # )
                    
                    F_inter = np.dot(np.dot(inv(eig_inv).T, F_inter), inv(eig_inv))
                    print("F_inter:")
                    print(F_inter)
                    print("F_A:")
                    print(F)
                    xi = F_inter * 0
                    od_inds = []
                    self.total_off_diags_buff = (len(xi) ** 2 - len(xi)) / 2
                    
                    sym_sort = np.arange(0, len(TED_obj.TED))
                    sym_sort = np.array([sym_sort])
                    # print(sym_sort.T)
                    # raise RuntimeError

                    irreps_CMA0, freqs = self.mode_symmetry_sort(
                        diag_TED, sym_sort.T, diag_GF.freq, percent_tol=51.0
                    )
                   
                    print(sym_sort)
                    print(irreps_CMA0) 
                    print(np.array(irreps_CMA0).flatten()) 
                    print(freqs)
                    b_ind = np.array(irreps_CMA0).flatten()
                    # raise RuntimeError

                    for i in range(len(xi)):
                        for j in range(i + 1):
                            if i != j:
                                a = b_ind[i]
                                b = b_ind[j]
                                freq_a = diag_GF.freq[a]
                                freq_b = diag_GF.freq[b]
                                buff = np.abs(F_inter[i, j])
                                xi[i, j] = buff / np.sqrt(
                                    np.abs(F_inter[i, i])
                                    * np.abs(F_inter[j, j])
                                )
                                omega = 4 * xi[i, j]**2 * freq_a * freq_b
                                omega += (freq_a - freq_b)**2
                                omega = 0.5*np.abs(np.sqrt(omega) - np.abs(freq_a - freq_b))
                                
                                # print(i, j)
                                # print("Omega diagnostic in wavenumbers?")
                                # print(omega)
                                if omega > 10.0:
                                    print(i, j)
                                    print("Omega diagnostic in wavenumbers:")
                                    print(omega)
                                    print("Xi diagnostic:")
                                    print(xi[i, j])

                                # if xi[i, j] > xi_tol_i:
                                    # od_inds.append([i, j])
                    # raise RuntimeError
            
            else:
                print(
                    "Only CMA-(1-3) off_diag algorithms are implemented at the moment."
                )
                print("Please enter 1, 2, or 3 for the off_diag option.")
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
            upper_triang = abs(np.triu(FC, 1))
            length = len(upper_triang)
            for i in range(0, n):
                index = np.argmax(upper_triang)
                if index > length:
                    two_d = [index // length, index % length]
                else:
                    two_d = [0, index]
                indexes.append(two_d)

                upper_triang[two_d[0], two_d[1]] = 0
            return indexes

    # This function returns a sorted list of frequencies, with any non-coupling frequencies being deleted.
    def mode_symmetry_sort(self, TED, sym_sort, freqs, percent_tol=90.0):
        ref_TED_init = TED
        sym_modes = []
        for irrep in sym_sort:
            irrep_modes = []
            if len(np.array(irrep).shape) > 1:
                buff_irrep = np.array(irrep).flatten()
            else:
                buff_irrep = irrep
            for i in range(len(ref_TED_init.T)):
                Sum = 0
                for j in buff_irrep:
                    Sum += ref_TED_init.T[i, j]
                print(i)
                print(buff_irrep)
                print(Sum)
                if Sum > percent_tol:
                    irrep_modes.append(i)
            # print(np.array(irrep)+1)
            # print(np.array(irrep_modes)+1)
            if len(irrep_modes) != len(buff_irrep):
                print("Something's wrong with the irrep symmetry sorter:")
                raise RuntimeError
            sym_modes.append(irrep_modes)

        sym_freqs = copy.deepcopy(sym_modes)
        del_list = []
        print("Sym_modes:")
        print(sym_modes)
        for i in range(len(sym_modes)):
            if len(sym_modes[i]) > 0:
                # if len(sym_modes[i]) == 1:
                    # print("We made it here:")
                    # del_list.append(i)
                for j in range(len(sym_modes[i])):
                    # print(i, j)
                    index = sym_modes[i][j]
                    sym_freqs[i][j] = freqs[index].copy()
                sym_freqs[i].reverse()
            else:
                pass
        print(del_list)
        del_list.reverse()
        if len(del_list):
            print("These modes will be deleted.")
            for i in del_list:
                print(freqs[sym_modes[i][0]])
        for i in del_list:
            del sym_freqs[i]
        flat_sym_freqs = [x for xs in sym_freqs for x in xs]
        flat_sym_freqs = np.array(flat_sym_freqs)

        return sym_modes, flat_sym_freqs
