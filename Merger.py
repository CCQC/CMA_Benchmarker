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
from concordantmodes.cma import ConcordantModes
from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.gf_method import GFMethod
from concordantmodes.g_matrix import GMatrix
from concordantmodes.g_read import GrRead
from concordantmodes.int2cart import Int2Cart
from concordantmodes.options import Options
from concordantmodes.sapelo_template import SapeloTemplate
from concordantmodes.s_vectors import SVectors
from concordantmodes.symmetry import Symmetry
from concordantmodes.ted import TED
from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.zmat import Zmat

from fractions import Fraction

class Merger:
    """
    Driver class for the Concordand Modes Benchark
    """
    def __init__(self, cmaA_path=None):
        # TODO create preset bundles of options for different clusters
        # and Qchem programs. This applies for concordantmodes code as well.
        options_kwargs = {
            "queue" : "batch",
            "program_b" : "molpro",
            # likely the only relevant option keywords
            "time_limit" : "4:00:00",
            "cluster" : "slurm",
            "disp_b" : 0.02,
            "ted_check" : True,
        }
        options_obj = Options(**options_kwargs)
        self.options = options_obj
        self.cmaA_path = cmaA_path

    def run(
        self,
        opts,
        Proj,
        energy_regex=None,
        success_regex=None,
        cmaA_coord=None,
        sym_sort=None,
        coord_type_b="internal",
        xi_tol=None,
        omega_tol=None,
        od_inds=None,
        tiles=None,
        tile_type=None,
        tile_xi=None,
    ):
        xi_tol = [] if xi_tol is None else xi_tol
        omega_tol = [] if omega_tol is None else omega_tol
        od_inds = [] if od_inds is None else od_inds
        self.tiles = [] if tiles is None else tiles
        tile_type = [] if tile_type is None else tile_type
        tile_xi = {} if tile_xi is None else tile_xi

        self.coord_type_b = coord_type_b

        self.Proj = Proj

        self.options = opts
        rootdir = os.getcwd()
        
        zmat_obj = Zmat(self.options)
        zmat_obj.run()
        
        if self.options.geom_check:
            raise RuntimeError

        # Do we want to use molsym_symmetry or "manual" symmetry via sym_sort?
        self.symm_obj = Symmetry(zmat_obj, self.options, self.Proj)
        if self.options.molsym_symmetry:
            self.symm_obj.run()
        else:
            """
            We won't run the symmetry code, but we'll create a dummy object to be passed as an argument.
            #TODO: This is a hacky way to do this, but it's a quick fix for now. Maybe reincorporate symmetry as a s_vector obj?
            """
            self.symm_obj.dummy_obj()
            self.symm_obj.symtext = None
            # check if sym_sort object was passed in. If so, intialize sym_sort objects
            if len(sym_sort) > 1:
                self.symm_obj.create_flat_sym_sort(sym_sort)
                flat_sym_sort = self.symm_obj.flat_sym_sort
                flat_sym_sort_inv = self.symm_obj.flat_sym_sort_inv

        np.set_printoptions(edgeitems=60, linewidth=1000)

        b_dir = os.getcwd() + self.cmaA_path
        
        os.chdir(b_dir)
        
        self.options.energy_regex_b = energy_regex
        self.options.success_regex_b = success_regex
        self.cma = ConcordantModes(self.options, proj=self.Proj)
        self.cma.symm_obj = self.symm_obj
        self.cma.sym_sort = sym_sort
        

        # Change outdated gradient file names in benchmarker.
        if os.path.exists(b_dir + "/fc_cart.grad"):
            shutil.move(b_dir + "/fc_cart.grad",b_dir + "/fc_cart_b.grad")
        if os.path.exists(b_dir + "/fc_cart.dat"):
            shutil.move(b_dir + "/fc_cart.dat",b_dir + "/fc_cart_b.dat")
        if os.path.exists(b_dir + "/fc_int_nat.dat"):
            shutil.move(b_dir + "/fc_int_nat.dat",b_dir + "/fc_int_b.dat")
        if self.coord_type_b.lower() == 'cartesian':
            if os.path.exists(b_dir + "/fc_cart_b.grad") and os.path.exists(b_dir + "/fc_cart_b.dat") and os.path.exists(b_dir + "/DispsB"):
                shutil.rmtree(b_dir + "/DispsB")
        if self.coord_type_b.lower() == 'internal':
            if os.path.exists(b_dir + "/fc_int_b.dat") and os.path.exists(b_dir + "/DispsB"):
                shutil.rmtree(b_dir + "/DispsB")
        if os.path.exists(b_dir + "/DispsB"):
            self.options.gen_disps_b = False
            self.options.calc_b = False
        
        # Might need to go back and check that cartesians work.
        F_b, grad_b = self.cma.compute_hessian(
            "B",
            self.options.deriv_level_b,
            self.coord_type_b,
            zmat_obj,
            self.options,
            os.getcwd(),
            zmat_obj.cartesians_b,
            off_diag=0,
            prog=self.options.program_b,
            gen_disps=self.options.gen_disps_b,
            calc=self.options.calc_b,
        )

        os.chdir(rootdir)

        g_mat = GMatrix(zmat_obj, self.cma.s_vec, self.options, proj=self.Proj)
        g_mat.run()
        G = g_mat.G.copy()
        print(g_mat.G.shape)
        print(F_b.shape)
        
        self.options.init_bool = False
        
        if len(sym_sort) > 1:
            F_b, G = self.cma.symm_obj.GF_sym_sort(F_b, G, sym_sort)

        print("Level B Frequencies:")
        b_GF = GFMethod(
            G.copy(),
            F_b.copy(),
            zmat_obj,
            self.cma.TED_obj,
            self.options,
            symtext=self.cma.symm_obj.symtext,
        )
        b_GF.run()
        
        b_GF.ted.TED[np.abs(b_GF.ted.TED) < 1e-5] = 0.0
        ted_b = b_GF.ted.TED
        
        self.ref_b = b_GF.freq
        if len(sym_sort):
            self.irreps_b, flat_sym_freqs = self.cma.symm_obj.mode_symmetry_sort(
                ted_b, sym_sort, b_GF.freq
            )
            self.ref_b = np.array(flat_sym_freqs)
            #### this block could probably be moved inside the symmetry.py module?
            flat_sym_modes_b = [x for xs in self.irreps_b for x in xs]
            print(flat_sym_modes_b)
            del_list = []
            flat_sym_modes_b = np.delete(np.array(flat_sym_modes_b), del_list)
            ted_b = ted_b.T
            ted_b = ted_b[flat_sym_modes_b]
            ted_b = ted_b.T
            #### end of block that could probably be moved inside the symmetry.py module?
        
        if len(self.tiles):
            self.tiles_b, sorted_freqs = self.cma.symm_obj.mode_symmetry_sort(
                ted_b, self.tiles, self.ref_b, percent_tol=80.0
            )



        F_b = np.dot(np.dot(b_GF.L.T, F_b), b_GF.L)
        F_b[np.abs(F_b) < self.options.tol] = 0
        G = np.dot(np.dot(LA.inv(b_GF.L), G), LA.inv(b_GF.L).T)
        G[np.abs(G) < self.options.tol] = 0
        # Now for the TED check.
        if self.options.ted_check:

            print("TED Frequencies: Degeneracy x Irrep")
            TED_GF = GFMethod(
                G,
                F_b,
                zmat_obj,
                self.cma.TED_obj,
                self.options,
                self.cma.symm_obj.symtext,
            )
            TED_GF.run()

        proj_tol = 1.0e-3
        self.eig_inv = inv(b_GF.L)  # (Normal modes (Q) x Sym internals (S) )
        for i in range(len(self.eig_inv)):
            self.eig_inv[i] = self.eig_inv[i] / LA.norm(self.eig_inv[i])
            self.eig_inv[i][np.abs(self.eig_inv[i]) < np.max(np.abs(self.eig_inv[i])) * proj_tol] = 0

        # Now run the TZ force constant transformation
        zmat_obj2 = Zmat(self.options)
        print(os.getcwd())
        zmat_obj2.run(zmat_name="zmat2")

        self.options.man_proj = True

        s_vec = SVectors(zmat_obj2, self.options)#, zmat_obj2.variable_dictionary_b)
        s_vec.run(zmat_obj2.cartesians_b, True, proj=self.Proj)
        

        TED_obj = TED(s_vec.proj, zmat_obj2, self.options)

        g_mat = GMatrix(zmat_obj2, s_vec, self.options)
        g_mat.run()

        G = g_mat.G.copy()
        init_bool = False
        # if os.path.exists(rootdir + "/fc2.dat"):
        if os.path.exists(os.path.join(rootdir,"fc2.dat")):
            f_read_obj = FcRead("fc2.dat")
        # elif os.path.exists(rootdir + "/FCMFINAL2"):
        elif os.path.exists(os.path.join(rootdir,"FCMFINAL2")):
            f_read_obj = FcRead("FCMFINAL2")
        else:
            raise RuntimeError

        self.options.second_order = False

        f_read_obj.run()
        f_conv_obj = FcConv(
            f_read_obj.fc_mat,
            s_vec,
            zmat_obj2,
            "internal",
            False,
            self.Proj,
            self.options,
        )
        f_conv_obj.run()
        F = f_conv_obj.F

        G = np.dot(np.dot(self.Proj.T, G), self.Proj)
        # Conversion to aJ/Ang
        F_aJ = F.copy()
        F_aJ *= 4.3597447222071
        F_aJ /= 0.529177210903

        np.set_printoptions(edgeitems=60, linewidth=10000)

        if len(sym_sort) > 1:
            F, G = self.cma.symm_obj.GF_sym_sort(F, G, sym_sort)
       
        G = np.dot(np.dot(self.eig_inv, G), self.eig_inv.T)
        F = np.dot(np.dot(inv(self.eig_inv).T, F), inv(self.eig_inv))

        np.set_printoptions(precision=7, edgeitems=60, linewidth=10000)
        print("Normal Mode G")
        print(G)
        print("Normal Mode F")
        print(F)

        full_GF = GFMethod(G, F, zmat_obj2, TED_obj, self.options)
        full_GF.run()
        self.ted = full_GF.ted.TED  # TED matrix

        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" Full Hessian TED"))
        print("////////////////////////////////////////////")
        TED_obj.run(np.dot(b_GF.L, full_GF.L), full_GF.freq, rect_print=False)
        self.reference_freq = full_GF.freq
        if len(sym_sort):
            self.irreps_ref, flat_sym_freqs = self.cma.symm_obj.mode_symmetry_sort(
                TED_obj.TED, sym_sort, self.reference_freq
            )
            self.reference_freq = np.array(flat_sym_freqs)
        if len(self.tiles):
            self.tiles_ref, sorted_freqs = self.cma.symm_obj.mode_symmetry_sort(
                TED_obj.TED, self.tiles, self.reference_freq, percent_tol=80.0
            )

        self.reference_TED = TED_obj.TED
        ref_TED = self.reference_TED

        Fdiag = copy.copy(np.diag(np.diag(F)))

        print("Diagonal Force constant matrix in lower level normal mode basis:")
        print(Fdiag)
        diag_GF = GFMethod(G, Fdiag, zmat_obj2, TED_obj, self.options)
        
        diag_GF.run()
        
        self.Freq_CMA0 = diag_GF.freq
        diag_TED = diag_GF.ted.TED.copy()
        
        self.denom = len(self.Freq_CMA0) * 1.0
        freq_diff = self.Freq_CMA0 - full_GF.freq
        self.outliers = len(freq_diff[np.abs(freq_diff) > 2.5])
        pure_freq_diff = self.ref_b - full_GF.freq
        self.pure_outliers = len(pure_freq_diff[np.abs(pure_freq_diff) > 2.5])

        if self.options.coords == 'Delocalized':
            self.Freq_delocalized = diag_GF.freq
        
        # Print Diagonal TED here in projected basis

        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" CMA-0 TED"))
        print("////////////////////////////////////////////")
        TED_obj.run(np.dot(b_GF.L, diag_GF.L), diag_GF.freq, rect_print=False)
        if len(sym_sort):
            self.irreps_CMA0, flat_sym_freqs = self.cma.symm_obj.mode_symmetry_sort(
                TED_obj.TED, sym_sort, self.Freq_CMA0
            )
            self.Freq_CMA0 = np.array(flat_sym_freqs)
        if len(self.tiles):
            self.tiles_CMA0, sorted_freqs = self.cma.symm_obj.mode_symmetry_sort(
                TED_obj.TED, self.tiles, self.Freq_CMA0, percent_tol=80.0
            )
        
        # Run off-diags here.
        if not self.options.off_diag: 
            if len(self.options.other_F_matrix_del):
                F_inter = self._read_force_constant(
                    "inter_fc.dat",
                    "inter_fc_cart.dat",
                    "inter_fc_cart.grad",
                    zmat_obj2,
                    s_vec,
                )
                F_inter, _ = self.cma.symm_obj.GF_sym_sort(F_inter, G, sym_sort)
                F_inter = self._transform_force_constant(F_inter)
                F_inter_del = self._read_force_constant(
                    "inter_fc_del.dat",
                    "inter_fc_cart_del.dat",
                    "inter_fc_cart_del.grad",
                    zmat_obj2,
                    s_vec,
                )
                F_interi_del, _ = self.cma.symm_obj.GF_sym_sort(F_inter_del, G, sym_sort)
                print(F_inter_del)
                F_inter_del = self._transform_force_constant(F_inter_del)

                # F_delta = F_inter_del - F_inter
                F_delta = F_inter_del

                print(len(Fdiag))
                print(Fdiag)
                print("Adding on all off-diagonals:")
                for i in range(len(Fdiag)):
                    for j in range(i):
                        # print(i,j)
                        Fdiag[i,j] = F_delta[i,j]
                        Fdiag[j,i] = Fdiag[j,i]


                # temp = self._insert_off_diags(Fdiag, F_delta, od_inds)
                self.Freq_cmaA = self._run_cma(
                    G,
                    Fdiag,
                    zmat_obj2,
                    TED_obj,
                    b_GF,
                    sym_sort,
                    "1",
                )
                self.Freq_cma2 = np.array([self.Freq_cmaA])

            return

        elif self.options.off_diag == 1:
            print("Adding on these off-diagonals:")
            print(len(od_inds))
            print(od_inds)

            temp = self._insert_off_diags(Fdiag, F, od_inds)

            print("Time for some off-diags")
            self.Freq_cmaA = self._run_cma(
                G,
                temp,
                zmat_obj2,
                TED_obj,
                b_GF,
                sym_sort,
                "1",
            )
            return

        elif self.options.off_diag == 2:
            self.Freq_cma2 = np.array([])
            self.eta_num = np.array([])
            self.eta_denom = np.array([])
            self.total_off_diags = np.array([])

            F_inter = self._get_interaction_force_constant(
                zmat_obj2,
                s_vec,
            )

            for xi_tol_i in xi_tol:
                print(xi_tol_i)
                print("Time for some off-diags")
                self.xi_tol_i = xi_tol_i
                if F_inter is not None:
                    print("F_inter:")
                    print(F_inter)
                    print("F_A:")
                    print(F)

                    xi, od_inds = self._find_xi_off_diags(
                        F_inter,
                        sym_sort,
                        tile_xi,
                        tile_type,
                    )

                    self._record_off_diag_statistics(od_inds)

                    F = self._apply_auxiliary_force_constant(
                        zmat_obj2,
                        s_vec,
                        F,
                    )

                    temp = self._insert_off_diags(Fdiag, F, od_inds)
                    print("temp:")
                    print(temp)


                print(temp)
                print(F)

                frequencies = self._run_cma(
                    G,
                    temp,
                    zmat_obj2,
                    TED_obj,
                    b_GF,
                    sym_sort,
                    "2",
                )

                self.Freq_cma2 = np.append(
                    self.Freq_cma2,
                    frequencies,
                    axis=0,
                )

            self.Freq_cma2 = self.Freq_cma2.reshape(
                len(xi_tol),
                -1,
            )
            return

        elif self.options.off_diag == 3:
            self.Freq_cma3 = np.array([])
            self.eta_num = np.array([])
            self.eta_denom = np.array([])
            self.total_off_diags = np.array([])

            F_inter = self._get_interaction_force_constant(
                zmat_obj2,
                s_vec,
            )

            if F_inter is None:
                return

            print("F_inter:")
            print(F_inter)
            print("F_A:")
            print(F)

            diag_sym_sort = np.arange(len(TED_obj.TED))[None, :].T
            print(diag_TED)

            irreps_CMA0, _ = self.cma.symm_obj.mode_symmetry_sort(
                diag_TED,
                diag_sym_sort,
                diag_GF.freq,
                percent_tol=51.0,
            )

            b_ind = np.asarray(irreps_CMA0).flatten()

            for omega_tol_i in omega_tol:
                od_inds = self._find_omega_off_diags(
                    F_inter,
                    b_ind,
                    diag_GF.freq,
                    omega_tol_i,
                    sym_sort,
                )

                print("CMA3 off-diagonal elements:")
                print(od_inds)

                if len(od_inds) > self.total_off_diags_buff:
                    raise RuntimeError

                self._record_off_diag_statistics(od_inds)


                F = self._apply_auxiliary_force_constant(
                    zmat_obj2,
                    s_vec,
                    F,
                )

                temp = self._insert_off_diags(Fdiag, F, od_inds)
                
                print("temp vs F:")
                print(temp)
                print(F)

                frequencies = self._run_cma(
                    G,
                    temp,
                    zmat_obj2,
                    TED_obj,
                    b_GF,
                    sym_sort,
                    "3",
                )

                self.Freq_cma3 = np.append(
                    self.Freq_cma3,
                    frequencies,
                    axis=0,
                )

            self.Freq_cma3 = self.Freq_cma3.reshape(
                len(omega_tol),
                -1,
            )
            return

        print(
            "Only CMA-(1-3) off_diag algorithms are implemented at the moment."
        )
        print("Please enter 1, 2, or 3 for the off_diag option.")
        raise RuntimeError



    def _transform_force_constant(self, fc):
        """Transform a force-constant matrix into the CMA coordinate basis."""
        transform = inv(self.eig_inv)
        return transform.T @ fc @ transform


    def _read_force_constant(self, filename, cartesian_filename, gradient_filename,
                             zmat_obj2, s_vec):
        """Read and, when necessary, convert an auxiliary force-constant matrix."""
        if (
            self.coord_type_b == "internal"
            and os.path.exists(filename)
        ):
            reader = FcRead(filename)
            reader.run()
            return reader.fc_mat

        if (
            self.coord_type_b == "cartesian"
            and os.path.exists(cartesian_filename)
        ):
            fc_reader = FcRead(cartesian_filename)
            gr_reader = GrRead(gradient_filename)

            gr_reader.run(zmat_obj2.cartesians_b)
            fc_reader.run()

            converter = FcConv(
                fc_reader.fc_mat,
                s_vec,
                zmat_obj2,
                "internal",
                False,
                self.Proj,
                self.options,
            )
            converter.run(grad=gr_reader.cart_grad)
            return converter.F

        return None


    def _get_interaction_force_constant(self, zmat_obj2, s_vec):
        """Read and transform the off-diagonal force-constant matrix."""
        if not len(self.options.other_F_matrix):
            return None

        fc = self._read_force_constant(
            "inter_fc.dat",
            "inter_fc_cart.dat",
            "inter_fc_cart.grad",
            zmat_obj2,
            s_vec,
        )

        return None if fc is None else self._transform_force_constant(fc)


    def _get_auxiliary_force_constant(self, zmat_obj2, s_vec, F):
        """Read and transform the optional auxiliary force-constant matrix."""
        if not len(self.options.aux_F):
            return F

        fc = self._read_force_constant(
            "aux_fc.dat",
            "aux_fc_cart.dat",
            "aux_fc_cart.grad",
            zmat_obj2,
            s_vec,
        )

        return F if fc is None else self._transform_force_constant(fc)


    def _tile_index(self, mode):
        """Return the tile containing a mode."""
        for index, tile in enumerate(self.tiles_b):
            if mode in tile:
                return index
        raise RuntimeError(
            f"Could not find mode {mode} in self.tiles_b."
        )


    def _xi_threshold(self, mode_a, mode_b, tile_xi, tile_type):
        """Return the xi threshold for a pair of modes."""
        if not len(self.tiles):
            return None

        tile_a = self._tile_index(mode_a)
        tile_b = self._tile_index(mode_b)
        type_a = tile_type[tile_a]
        type_b = tile_type[tile_b]

        if type_a == "i" or type_b == "i":
            if tile_a == tile_b:
                return tile_xi["ii"]
            if type_a == "i" and type_b == "i":
                return tile_xi["i1i2"]
            return tile_xi["mi"]

        if type_a in ("m1", "m2") and type_b in ("m1", "m2"):
            if tile_a != tile_b:
                return tile_xi["m1m2"]
            return tile_xi[f"{type_a}{type_a}"]

        print(
            "Tile type must be m1, m2, or i, check what you put in the tile_type array"
        )
        print(type_a)
        print(type_b)
        raise RuntimeError


    @staticmethod
    def _xi_value(F_inter, a, b):
        """Return the normalized off-diagonal coupling between two modes."""
        return abs(F_inter[a, b]) / np.sqrt(
            abs(F_inter[a, a]) * abs(F_inter[b, b])
        )


    def _candidate_pairs(self, irreps, n_modes, use_symmetry):
        """Yield mode pairs, optionally restricted to symmetry-equivalent modes."""
        if use_symmetry:
            for irrep in irreps:
                for i in range(len(irrep)):
                    for j in range(i):
                        yield irrep[i], irrep[j]
            return

        for i in range(n_modes):
            for j in range(i):
                yield i, j


    def _find_xi_off_diags(self, F_inter, sym_sort, tile_xi, tile_type):
        """Find CMA-2 off-diagonal elements using the xi criterion."""
        xi = np.zeros_like(F_inter)
        od_inds = []

        use_symmetry = len(sym_sort) > 1
        irreps = self.irreps_b if use_symmetry else [range(len(F_inter))]

        if use_symmetry:
            self.total_off_diags_buff = sum(
                (len(irrep) ** 2 - len(irrep)) / 2
                for irrep in irreps
            )
        else:
            self.total_off_diags_buff = (len(F_inter) ** 2 - len(F_inter)) / 2

        for a, b in self._candidate_pairs(
            irreps,
            len(F_inter),
            use_symmetry,
        ):
            threshold = (
                self._xi_threshold(a, b, tile_xi, tile_type)
                if len(self.tiles)
                else self.xi_tol_i
            )
            coupling = self._xi_value(F_inter, a, b)
            xi[a, b] = coupling

            # With no tile information, retain the original behavior by
            # requiring the caller to provide the applicable threshold.
            if threshold is not None and coupling > threshold:
                od_inds.append([a, b])
        return xi, od_inds


    @staticmethod
    def _omega_value(xi, freq_a, freq_b):
        """Calculate the CMA-3 omega diagnostic."""
        value = 4 * xi**2 * freq_a * freq_b
        value += (freq_a - freq_b) ** 2
        return 0.5 * abs(np.sqrt(value) - abs(freq_a - freq_b))


    def _find_omega_off_diags(self, F_inter, b_ind, diag_freq, omega_tol_i,
                              sym_sort):
        """Find CMA-3 off-diagonal elements using the omega criterion."""
        xi = np.zeros_like(F_inter)
        od_inds = []

        if len(sym_sort) > 1:
            irreps = self.irreps_b
            self.total_off_diags_buff = sum(
                (len(irrep) ** 2 - len(irrep)) / 2
                for irrep in irreps
            )

            pairs = (
                (irrep[i], irrep[j], b_ind[irrep[i]], b_ind[irrep[j]])
                for irrep in irreps
                for i in range(len(irrep))
                for j in range(i)
            )
        else:
            self.total_off_diags_buff = (
                len(F_inter) ** 2 - len(F_inter)
            ) / 2
            pairs = (
                (i, j, b_ind[i], b_ind[j])
                for i in range(len(F_inter))
                for j in range(i)
            )

        for a, b, freq_a_ind, freq_b_ind in pairs:
            coupling = self._xi_value(F_inter, a, b)
            xi[a, b] = coupling

            omega = self._omega_value(
                coupling,
                diag_freq[freq_a_ind],
                diag_freq[freq_b_ind],
            )

            if omega > omega_tol_i:
                print(a, b)
                print("Omega diagnostic in wavenumbers:")
                print(omega)
                print("Xi diagnostic:")
                print(xi[a, b])
                od_inds.append([a, b])

        return od_inds


    @staticmethod
    def _insert_off_diags(matrix, source, od_inds):
        """Copy selected off-diagonal elements symmetrically."""
        result = copy.copy(matrix)

        for a, b in od_inds:
            element = source[a, b]
            print(a, b)
            print(element)
            result[a, b] = element
            result[b, a] = element

        return result


    def _run_cma(self, G, F_matrix, zmat_obj2, TED_obj, b_GF,
                 sym_sort, cma_number):
        """Run GF/TED for one CMA force-constant matrix."""
        gf = GFMethod(
            G,
            F_matrix,
            zmat_obj2,
            TED_obj,
            self.options,
        )
        gf.run()
        frequencies = gf.freq.copy()

        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(f" CMA-{cma_number} TED"))
        print("////////////////////////////////////////////")

        TED_obj.run(
            b_GF.L @ gf.L,
            gf.freq,
            rect_print=False,
        )

        if len(sym_sort):
            irreps, sorted_freqs = self.cma.symm_obj.mode_symmetry_sort(
                TED_obj.TED,
                sym_sort,
                frequencies,
            )
            setattr(self, f"irreps_CMA{cma_number}", irreps)
            frequencies = np.array(sorted_freqs)

        return frequencies


    def _record_off_diag_statistics(self, od_inds):
        """Update CMA off-diagonal and eta statistics."""
        print("CMA{} off-diagonal elements:".format(self.options.off_diag))
        print(od_inds)
        print(len(od_inds))
        print(self.total_off_diags_buff)
        print("% ODs")
        print(len(od_inds) / self.total_off_diags_buff * 100)
        print("% eta")
        print(len(od_inds) / len(self.Freq_CMA0) * 100.0)

        self.cma_off_diags = len(od_inds)
        self.total_off_diags = np.append(
            self.total_off_diags,
            self.total_off_diags_buff,
        )
        self.eta_num = np.append(
            self.eta_num,
            float(self.cma_off_diags),
        )
        self.eta_denom = np.append(
            self.eta_denom,
            float(len(self.Freq_CMA0)),
        )


    def _apply_auxiliary_force_constant(self, zmat_obj2, s_vec, F):
        """Replace F with the transformed auxiliary force constant when present."""
        return self._get_auxiliary_force_constant(zmat_obj2, s_vec, F)

        
