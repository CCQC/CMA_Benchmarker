import shutil as sh
import os
import sys

here = os.getcwd()
dirs = os.listdir(here)

for i in dirs:
    if os.path.isdir(os.getcwd() + "/" + i):
        print(i)
        print(i.split("_"))
        namearr = i.split("_")
        if len(namearr) > 2:
            new_name = namearr[0] + "_" + namearr[1]
            for j in range(len(namearr) - 2):
                new_name += "\\ " + namearr[j+2]
            print("This is the new name")
            print(new_name)
            os.rename(os.path.join(here,i),os.path.join(here,new_name))


