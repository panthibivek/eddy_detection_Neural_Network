#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 09:12:02 2022

@author: bivek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 14:57:00 2021

@author: bivek
"""

#%matplotlib inline

import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
import scipy.spatial.qhull as qhull
import matplotlib.tri as mtri
import matplotlib.cm as cm
xr.set_options(display_style="text")

def comparison_function(data_address, grid_address, day=0):
    data = xr.open_dataset(data_address)
    grid = xr.open_dataset(grid_address)
    data_sample = data.ssh[day,:].values
    model_lon = grid.lon.values
    model_lat = grid.lat.values

    triangularized_data = data_triangulation(data, grid, data_sample, model_lon, model_lon)
    
    while(1):
        lon, lat = region_init()
        lon2, lat2 = np.meshgrid(lon, lat)
    
        nearest_nei_interpolator(triangularized_data(lon2, lat2), data_sample, model_lon, model_lat, lon2, lat2)
        linear_interpolator(triangularized_data(lon2, lat2))
    
def region_init(res: float = 1/12, ):
    flag_input = 1
    #defining our region
    print("Please input values in degree")
    print("Please enter with '-' sign if the values are in degree south or west")

    while(flag_input):
        left = input("left: ")
        right = input("right: ")
        top = input("top: ")
        bottom = input("bottom: ")
        try: 
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
            flag_input = 0
        except ValueError:
            print("Please enter only integer values")
    lon = np.arange(left, right, res)
    lat = np.arange(top, bottom, res)
    return lon, lat

def graph(data):
    plt.figure(figsize=(10,10))
    plt.imshow(np.flipud(data), cmap=cm.seismic, vmin=-1.5, vmax=0.5)
    plt.colorbar(orientation='horizontal', pad=0.04)


def triangulator(data, elements, model_lon, model_lat):
    #elements = (grid.elements.data.astype('int32') - 1).T
    d = model_lon[elements].max(axis=1) - model_lon[elements].min(axis=1)
    no_cyclic_elem = np.argwhere(d < 100).ravel()
    triang = mtri.Triangulation(model_lon, model_lat, elements[no_cyclic_elem])
    tri = triang.get_trifinder()
    return triang, tri
    
def data_triangulation(triang, tri, data_sample):
    triangularized_data = mtri.LinearTriInterpolator(triang, data_sample,trifinder=tri)
    return triangularized_data
    
def nearest_nei_interpolator(masked_data, data_sample, model_lon, model_lat, lon2, lat2):   
    #fesom
    points = np.vstack((model_lon, model_lat)).T
    nn_interpolation = NearestNDInterpolator(points, data_sample)
    interpolated_nn_fesom = nn_interpolation((lon2, lat2))
    mask = masked_data.mask
    interpolated_nn_fesom_masked = np.ma.array(interpolated_nn_fesom, mask=mask)
    return interpolated_nn_fesom_masked
    
def linear_interpolator(masked_data):   
    graph(masked_data)
    


