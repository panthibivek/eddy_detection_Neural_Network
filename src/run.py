#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:03:28 2022

@author: bivek
"""
import os
import sys

from function_03 import*

input_path_data = sys.argv[1]
input_path_grid = sys.argv[2]
input_path_nod2dfile = sys.argv[3]
input_path_elm2dfile = sys.argv[4]
year = sys.argv[5]
month = sys.argv[6]
input_left = sys.argv[7]
input_right = sys.argv[8]
input_top = sys.argv[9]
input_bottom = sys.argv[10]
input_nc_flag = sys.argv[11]


#checking if the path address is valid or not
if not os.path.exists(input_path_data):
    print(input_path_data, "doesn't exist!")
    sys.exit(1)
if not os.path.exists(input_path_grid):
    print(input_path_grid, "doesn't exist!")
    sys.exit(1)
if not os.path.exists(input_path_nod2dfile):
    print(input_path_nod2dfile, "doesn't exist!")
    sys.exit(1)
if not os.path.exists(input_path_elm2dfile):
    print(input_path_elm2dfile, "doesn't exist!")
    sys.exit(1)

interpolator_object = Interpolator(input_path_data, input_path_grid, input_path_nod2dfile, input_path_elm2dfile, year, month, input_left, input_right, input_top, input_bottom, int(input_nc_flag))
#interpolator_object.linear_interpolation_action()
interpolator_object.nn_interpolation_action(1)
