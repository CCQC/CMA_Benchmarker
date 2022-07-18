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
from concordantmodes.int2cart import Int2Cart
from concordantmodes.options import Options
from concordantmodes.reap import Reap
from concordantmodes.s_vectors import SVectors
from concordantmodes.submit import Submit
from concordantmodes.ted import TED
from concordantmodes.trans_disp import TransDisp
from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.zmat import Zmat
import copy

np.set_printoptions(precision=2)

class Analysis(object):


    def __init__(self):
        print('Nothing to init')
    
    def run(self, TED_ll_normalcoord, ll_freqs, ll_modes,cma23_scaling):
    #def run(self, TED_ll_normalcoord, scaling):

        def mode_overlap(ll_modes):
            ll_modes_abs = abs(ll_modes)
            overlap = np.dot(ll_modes_abs.T,ll_modes_abs)
            return overlap
      
        def prediction(overlap, ll_freqs):
            diagnostic = np.zeros((len(ll_freqs),len(ll_freqs)))
            for i, x in enumerate(ll_freqs):
                for j, y in enumerate(ll_freqs):
                    if i != j:
                        diagnostic[i,j] = overlap[i,j]/(abs(x-y))
                    else:
                        continue 
            return diagnostic

        
        def checkted(ted, tol):
            temps = []
            tempvals = []
            counts = 0
            ted = abs(np.triu(ted,1))
            for i in range(0,np.shape(ted)[0]):
                ted_slice = ted[:,i] 
                temp = copy.copy(ted_slice)
                count = 0
                for j in range(0,np.shape(ted)[0]):
                    if i != j:
                        if temp[j] > tol:
                            #print(temp[j],'save this index', [i,j])
                            temps.append([i,j])
                            tempvals.append(temp[j]) 
                            count +=1
                counts += count
                offdiags = counts - np.shape(ted)[0]
                scaling = offdiags/np.shape(ted)[0]
                scaling = counts/np.shape(ted)[0]
                #if count > 1:
                #    print(f'There were {count -1} elements larger than {tol} for this mode')
                #else:
                #    print(f'The TED analysis has determined that no other mode mixes with it given the tolerance of {tol}')  
            
            #print(f'The total number of off-diagonals selected was {counts - np.shape(ted)[0]}')
            #print(f'If all of these are comptuted, this variant of CMA will be {scaling} x the cost of CMA0')
            #print(len(temps)) 
            #print(tempvals)
            #print(temps)
            temps = np.array(temps) 
            #temps = temps[:,np.argsort(tempvals)]
            temps = temps[np.argsort(tempvals)]
            #print(temps)
            temps = np.flip(temps)
            #print(temps) 
            return temps

        def n_largest(n, diagnostic):
            indexes = []
            upper_triang = abs(np.triu(diagnostic,1))
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
       
        def stats(data):
            hist, bin_edges = np.histogram(data) 
            return None
        
        #self.temps = checkted(TED_ll_normalcoord,0.09) 
        self.temps = checkted(TED_ll_normalcoord,0.04) 
        print('Length of cheating TED check, and the indices')
        print(len(self.temps))
        print(self.temps)
        
        #n = abs(len(self.temps) - len(ll_freqs)) 
        
        n = int(cma23_scaling*len(self.temps) - len(self.temps))
        print(f'this is n {n}') 
        overlap = mode_overlap(ll_modes)
        #print("overlap")
        #print(overlap)
        
        diagnostic = prediction(overlap, ll_freqs)
        
        print('This is the diagnostic')
        print(diagnostic)
        
        self.indexes = n_largest(n, diagnostic)
        print(f'{self.temps} largest true ted values')
        print(f'{n} largest diagnostic values')
        print(self.indexes)
        
        #self.indexes are the indices chosen using the overlap diagnostic, turn off for now
        #self.indexes = None 
        return self.temps, self.indexes

    
