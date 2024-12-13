import shutil as sh
import os
import sys

here = os.getcwd()
dirs = os.listdir(here)

for i in dirs:
    if os.path.isdir(os.getcwd() + "/" + i):
        # print(i.split("\\\ "))
        namearr = i.split("\\ ")
        if len(namearr) > 1:
            print(i)
            new_name = namearr[0] + "_" + namearr[1]
            print("This is the new name")
            print(new_name)
            os.rename(os.path.join(here,i),os.path.join(here,new_name))


