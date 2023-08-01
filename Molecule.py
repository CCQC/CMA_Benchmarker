# Class for collecting and processing CMA benchmarking data 
# As well as generating LaTeX-formatted SI on a molecule by molecule basis.

import os
import re

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 15)

class Molecule(object):

    def __init__(self,job):
        info = re.search(r"/(\d*)_.*/(\d*)_(.*)/", job)
        self.ID = f"{info.group(1)}.{info.group(2)}"
        self.name = info.group(3)
        if info.group(1) == "1":
            self.section = "\\section{Closed Shell}\n\n"
        elif info.group(1) == "2":
            self.section = "\\section{Open Shell}\n\n"
        self.geoms = {}
        self.freqs = {}
        self.ted = {}
        self.proj = None
    
    def run(self):
        # ID info
        print('Molecule Class!')
        print("="*25)
        print(f"{self.name + ' ' + self.ID:^25}")
        print("="*25)

        # Geometries
        print("Geometries:")
        print("-----------")
        for geom in self.geoms:
            print(geom)
            for line in self.geoms[geom]:
                print(f"{line[0]:<2}    {float(line[1]):>13.10f}    {float(line[2]):>13.10f}    {float(line[3]):>13.10f}")
            print()

        # Frequencies
        print("Frequencies:")
        print("------------")
        # Creat multiline index
        t = [(head.split()[0], head.split()[1]) for head in self.freqs.keys()] 
        ind = pd.MultiIndex.from_tuples(t)
        df = pd.DataFrame(data=self.freqs)
        df.columns = ind
        print(df.to_string(index=False, float_format="%.2f"))
        print()

        # Nattys
        print("Nattys:")
        print("-------")
        for i, nic in enumerate(self.nics):
            if nic[0] > 1:
                print(f"{i+1:2}    1/\u221a{nic[0]}[", end='')
                for j, term in enumerate(nic[1:]):
                    out = term[1].coord_out
                    if j == 0:
                        if term[0] == 1:
                            print(out, end='')
                        elif term[0] == -1:
                            print(f"-{out}", end='')
                        else:
                            print(f"{term[0]}{out}", end='')
                        continue
                    if term[0] > 0:
                        sign = "+"
                    else:
                        sign = "-"
                    if abs(term[0]) == 1:
                        print(f" {sign} {out}", end='')
                    else:
                        print(f" {sign} {abs(term[0])}{out}", end='')
                print("]")
            else:
                print(f"{i+1:2}    {nic[1][1].coord_out}")
        print()
        
        # TED
        # print("TEDs:")
        # print("-----")
        # for ted in self.ted:
        #     print(ted)
        #     print("-"*(13+10*len(self.ted[ted])))
        #     print("     Freq    ", end="")
        #     for freq in self.freqs[f"Natty ({ted})"]:
        #         print(f"{freq:>6.1f}    ", end="")
        #     print("\n"+"-"*(13+10*len(self.ted[ted])))
        #     for i, line in enumerate(self.ted[ted]):
        #         print(f"Mode #{i+1:>3}    ", end="")
        #         for val in line:
        #             print(f"{val:>6.1f}    ", end="")
        #         print()
        #     print("-"*(13+10*len(self.ted[ted]))+"\n")

    def get_geoms(self, combo):
        cma1 = False
        if combo[0] not in os.listdir():
            cma1 = True
        for lvl in combo:
            if lvl in self.geoms.keys():
                if cma1 == True:
                    break
                else:
                    continue
            if cma1 == True:
                filename = f"./zmat"
            else:
                filename = f"./{lvl}/zmat"
            try: 
                with open(filename, "r") as f:
                    txt = f.readlines()
                self.direc_complete = True 
            except:
                print(f'There is no file named {filename} in this directory')
                self.direc_complete = False
                break
                #self.geoms = None 
            cart = []
            read = False
            for line in txt:
                if 'cart begin' in line:
                    read = True
                    continue
                elif 'cart end' in line:
                    break

                if read == True:
                    cart.append(line.split())
            self.geoms[lvl] = cart
            if cma1 == True:
                break

    def get_nattys(self,combo):
        # Get zmat information
        with open(f"./{combo[0]}/zmat", "r") as f:
            txt = f.readlines()

        zmat = []
        read = False
        for line in txt:
            if "ZMAT begin" in line:
                read = True
                continue
            elif "ZMAT end" in line:
                read = False
                break
            if re.search(r"\d+", line):
                zmat.append(line.split())

        # Format zmat to be more readable
        for i, coord in enumerate(zmat):
            zmat[i] = ZCoord(coord)
        
        # Combine proj and zmat 
        nics = []
        # print(self.proj.T)
        for coord in self.proj.T:
            tmp = coord/np.abs(coord)[coord!=0].min()
            tmp = np.rint(tmp)
            nic = [int(np.sum(np.square(tmp)))]
            for i, coeff in enumerate(tmp):
                if abs(coeff) > 0:
                    nic.append((int(coeff), zmat[i]))
            nics.append(nic)

        self.nics = nics

    def build_latex_output(self,cma1=False):
        txt = f"\\subsection{{\ \ \ \\ce{{{self.name}}}}}\n\n"

        # Geometries
        # txt += ("\\begin{table}[h!]\n"
        txt += ("\\subsubsection*{Geometries}\n")
                # "\\begin{table}[h!]\n"
                # "\\subsubsection*{Geometries}\n"
                # "\\begin{multicols}{2}\n"
                # "\\centering\n") 
        for i, g in enumerate(self.geoms):
            txt += ("\\begin{table}[h!]\n"
                    "\\centering\n")
            if g == "CCSD_T_TZ":
                cap = "CCSD(T)/cc-pVTZ"
            elif g == "CCSD_T_DZ":
                cap = "CCSD(T)/cc-pVDZ"
            elif g == "B3LYP_6-31G_2df,p_":
                cap = "B3LYP/6-31G(2df,p)"
            else:
                cap = g
    
            txt += (f"\\caption{{{cap} Cartesian Coordinates (Bohr)}}\n"
                    "\\begin{tabular}{llrrr}\n")
                    # "\\begin{tabular}{llrrr}\n"
            # txt += ("\\begin{tabular}{llrrr}\n"
                    # f"\\caption{{{cap} Cartesian Coordinates (Bohr)}}\n"
                     # "\\hline\n")
                     # "\\vrule\n")
                     # "\\toprule\n")
            for j, line in enumerate(self.geoms[g]):
                txt += f"{j+1:<2} & {line[0]:<2} & ${float(line[1]):>11.8f}$ & ${float(line[2]):>11.8f}$ & ${float(line[3]):>11.8f}$ \\\\\n"
            # txt += ("\\bottomrule\n"
            txt += ("\\end{tabular}\n")
            # txt += ("\\vrule\n"
            # txt += ("\\hline\n"
                    # "\\end{tabular}\n")
            # if i%2 == 1:
                # txt += ("\\end{multicols}\n"
                # txt += ("\\end{table}\n")
                # txt += ("\\end{table}\n\n"
                        # "\\end{table}\n\n"
                        # "\\begin{table}[h]\n")
                # if i < len(self.geoms)-2:
                    # txt += ("\\begin{multicols}{2}\n")
                # txt += "\\centering\n"
        
            txt += "\\end{table}\n\n"

        txt += "\\clearpage\n\n"
        # Frequencies
        # txt += ("\\begin{table}[h!]\n"
        txt += ("\\subsubsection*{Frequencies}\n"
                "\\begin{table}[h!]\n"
                # "\\subsubsection*{Frequencies}\n"
                "\\centering\n")
        
        labels = [("Reference","CCSD(T)/","cc-pVTZ")]
        for key in self.freqs:
            if key == "Ref (CCSD_T_DZ)":
                labels.append(("Reference","CCSD(T)/","cc-pVDZ"))
            if key == "Ref (B3LYP_6-31G_2df,p_)":
                labels.append(("Reference","B3LYP/","6-31G(2df,p)"))
            if key == "Natty (CCSD_T_DZ)":
                # labels.append(("CMA0","NCs","TZ/DZ"))
                if cma1:
                    labels.append(("CMA-0A","NCs","TZ/DZ"))
                else:
                    labels.append(("CMA-0B","NCs","TZ/DZ"))
            if key == "Red (CCSD_T_DZ)":
                # labels.append(("CMA0","DCs","TZ/DZ"))
                if cma1:
                    labels.append(("CMA-0A","DCs","TZ/DZ"))
                else:
                    labels.append(("CMA-0B","DCs","TZ/DZ"))
            if key == "Natty (B3LYP_6-31G_2df,p_)":
                # labels.append(("CMA0","NCs","TZ/DFT"))
                if cma1:
                    labels.append(("CMA-0A","NCs","TZ/DFT"))
                else:
                    labels.append(("CMA-0B","NCs","TZ/DFT"))
            if key == "Red (B3LYP_6-31G_2df,p_)":
                # labels.append(("CMA0","DCs","TZ/DFT"))
                if cma1:
                    labels.append(("CMA-0A","DCs","TZ/DFT"))
                else:
                    labels.append(("CMA-0B","DCs","TZ/DFT"))
        ind = pd.MultiIndex.from_tuples(labels)       
        fdf = pd.DataFrame(data=self.freqs)
        fdf.columns = ind
        fdf.index += 1
        
        txt += "\\caption{Harmonic frequencies for reference and CMA0 data.}\n"
        # if cma1:
            # txt += "\\caption{Harmonic frequencies for reference and CMA1 data.}\n"
        # else:
            # txt += "\\caption{Harmonic frequencies for reference and CMA0 data.}\n"
        txt += fdf.to_latex(float_format="%.2f",column_format="c"*(len(self.freqs)+1),multicolumn_format="c")
        txt += "\\end{table}\n\n"

        # Nattys
        tmp = ""
        for i, nic in enumerate(self.nics):
            tmp += f"  {i+1:<3}"
            for j, term in enumerate(nic[1:]):
                out = term[1].latex_out
                if j == 0:
                    if term[0] == 1:
                        tmp += f" & ${out}"
                    elif term[0] == -1:
                        tmp += f" & $-{out}"
                    else:
                        tmp += f" & ${term[0]}{out}"
                    continue
                if term[0] > 0:
                    sign = "+"
                else:
                    sign = "-"
                if abs(term[0]) == 1:
                    tmp += f" {sign} {out}"
                else:
                    tmp += f" {sign} {abs(term[0])}{out}"
                if (j % 10) + 1 == 10:
                    tmp += ("$ \\\\\n"
                            " & $")
            tmp += "$ \\\\\n"

        txt += "\\clearpage\n\n"
        # txt += ("\\begin{table}[h!]\n"
        txt += ("\\subsubsection*{Natural Internal Coordinates}\n"
                "\\begin{table}[h!]\n"
                # "\\subsubsection*{Natural Internal Coordinates}\n"
                "\\centering\n"
               f"\\caption{{Symmetrized, unnormalized natural internal coordinates for \\ce{{{self.name}}}.}}\n")

        if len(tmp) >= 40:
            txt += "\\small\n"

        txt += ("\\begin{tabular}{ll}\n")
        # txt += ("\\begin{tabular}{ll}\n"
                # "\\hline\n")
                # "\\vrule\n")
                # "\\toprule\n")

        txt += tmp

        # txt += ("\\bottomrule\n"
        # txt += ("\\vrule\n"
        # txt += ("\\hline\n"
                # "\\end{tabular}\n"
        txt += ("\\end{tabular}\n"
                "\\end{table}\n\n")

        # TED
        # txt += ("\\begin{table}\n"
                # "\\subsection*{Total Energy Distribution}\n"
                # "\\centering")

        # for ted in self.ted:
            # if ted[1] == "CCSD_T_DZ":
                # l = "CCSD(T)/cc-pVDZ"
            # elif ted[1] == "B3LYP_6-31G_2df,p_":
                # l = "B3LYLP/6-31G(2df,p)"
            # txt += f"\\caption{{Total energy distribution for CCSD(T)/cc-pVTZ frequencies on the {l} modes.}}\n"
            # tdf = pd.DataFrame(data=self.ted[ted])
            # tdf.columns = [f"{i:.1f}" for i in self.freqs[f"Natty ({ted[1]})"]]
            # tdf.index += 1
            # txt += tdf.to_latex(float_format="%.1f", column_format="l"+"r"*len(tdf.columns))

        # txt += "\\end{table}\n\n"

        txt += "\\clearpage\n\n"
        return txt

class ZCoord(object):
    '''
    Easy formating and output for zmat coordinates because they are annoying
    '''

    def __init__(self,ind):
        # Identify coordinate type
        if len(ind) == 2:
            self.ind = ",".join(ind)
            self.coord_out = f"r({self.ind})"
            self.latex_out = f"r_{{{self.ind}}}"
        if len(ind) == 3:
            self.ind = ",".join(ind)
            self.coord_out = f"\u03b8({self.ind})"
            self.latex_out = f"\\phi_{{{self.ind}}}"
        if len(ind) == 5:
            if ind[-1] == "T":
                self.ind = ",".join(ind[:-1])
                self.coord_out = f"\u03c4({self.ind})"
                self.latex_out = f"\\tau_{{{self.ind}}}"
            # Out of plane angles
            if ind[-1] == "O":
                self.ind = ",".join(ind[:-1])
                self.coord_out = f"\u03b3({self.ind})"
                self.latex_out = f"\\gamma_{{{self.ind}}}"
            # Linear bends
            if ind[-1] == "L":
                self.ind = ",".join(ind[:-1])
                self.coord_out = f"\u03b8({self.ind})"
                self.latex_out = f"\\theta_{{{self.ind}}}"
            # Linx bends
            if ind[-1] == "Lx":
                self.ind = ",".join(ind[:-1])
                self.coord_out = f"\u03b8x({self.ind})"
                self.latex_out = f"\\alpha^x_{{{self.ind}}}"
            # Liny bends
            if ind[-1] == "Ly":
                self.ind = ",".join(ind[:-1])
                self.coord_out = f"\u03b8y({self.ind})"
                self.latex_out = f"\\alpha^y_{{{self.ind}}}"


