#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:41:03 2021

@author: matthew
"""

#%%


def plot_gacos_data(r3_ma, dem_ma, plot_args, title):
    """ To explore how GACOS data relates to the DEM.  
    Inputs:
        r3_ma | rank 3 masked array | the gacos data, masked with the same mask as the DEM.  n_images x ny x nx
        dem_ma | rank 2 masked array | the DEM.  
        n_plots | rank 1 array | which data to plot.  E.g. np.arange(0,10) to plot the first 10.  
        
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    from scipy import optimize
    
    def linear_func(x, m, c):
        """ a sets amplitude, b sets frequency, c sets gradient of linear term
        """
        y = (m*x) + c
        return y
    
    n_cols = plot_args.shape[0]

    f, axes = plt.subplots(2,n_cols, figsize = (10,6))
    f.canvas.set_window_title(title)
    f.suptitle(title)
    
    data_to_plot = r3_ma[plot_args,]
    vmin = ma.min(data_to_plot)
    vmax = ma.max(data_to_plot)
    
    
    for n_col in range(n_cols):
        im = axes[0, n_col].imshow(r3_ma[n_col,], vmin = vmin, vmax = vmax)
        #plt.colorbar(im, ax=axes[0, n_col])
        
        axes[1, n_col].scatter(ma.compressed(dem_ma), ma.compressed(r3_ma[n_col,]))
        axes[1, n_col].set_xlabel('DEM')
        axes[1, n_col].set_ylim([vmin, vmax])
        axes[1, n_col].grid(True)
        params, params_covariance = optimize.curve_fit(linear_func, ma.compressed(dem_ma), ma.compressed(r3_ma[n_col,]))
        
        line_yvals = (params[0] * ma.compressed(dem_ma)) + params[1]
        axes[1, n_col].plot(ma.compressed(dem_ma), line_yvals, c= 'k')
        
        delay_0_height = (- params[1]) / params[0]
        
        axes[1, n_col].set_title(f"{params[0]} \n {params[1]} \n {int(delay_0_height)} (m)")    





#%%

def gacos_data_to_differences(gacos_data, acq_dates, ifg_dates):
    """ Given gacos data for certain acquisition dates, and a list of ifg_dates, calculate the gacos differences for the ifg dates.  
    Inputs:
        gacos_data | numpy rank 3 array | first dimension is acquistion number
        acq_dates | list of strings | date for each data in gacos_data, YYYYMMDD
        ifg_dates | list of strings | date range for each interferogram, YYYYMMDD_YYYYMMDD
    """
    import numpy as np
    
    gacos_differences = np.zeros((len(ifg_dates), gacos_data.shape[1], gacos_data.shape[2]))
    
    for ifg_n, incremental_ifg in enumerate(ifg_dates):
        master_date = incremental_ifg[9:]
        slave_date = incremental_ifg[:8]
        
        gacos_master = gacos_data[acq_dates.index(master_date),]
        gacos_slave = gacos_data[acq_dates.index(slave_date),]
        
        gacos_differences[ifg_n,] = gacos_master - gacos_slave
        
    return gacos_differences
    

#%%



def open_gacos_data(gacos_dir, lons_mg_new = None, lats_mg_new = None):
    """ Given a directory of GACOS files, return the data from these in a data cube, and some other auxiliary info (such as the lons and lats of each pixel.  )
    
    Inputs:
        gacos_dir | pathlib Path | directory that contains the unzippped gacos files (e.g. jpg, ztd, and rsc)
        lons_mg_new | numpy rank 2 | lons for all pixels in grid that gacos data should be resampled to (e.g. LiCSBAS grid)
        lats_mg_new | numpy rank 2 | lats for all pixels in grid that gacos data should be resampled to (e.g. LiCSBAS grid)
        
    Returns:
        gacos_datas | numpy rank 3 array | first dimension is acquistion number
        gacos_datas_resampled | numpy rank 3 array | Possibly return the data sampled to the new grid.  first dimension is acquistion number
        acq_dates | list | date for each data in gacos_data
        lons_mg | numpy rank 2 | lons for all pixels in a gacos image.  
        lats_mg | numpy rank 2 | lats for all pixels in a gacos image.  
        
    History:
        2021_08_19 | MEG | Written
        2021_08_25 | MEG | Add option to pass a new lon and lat grid to resample the GACOS data to (e.g. the grid that LiCSBAS is using.  )
        
    """
    import numpy as np
        
    def width_height_from_gacos_rsc(rsc_path):
        """ Given a gacos .rsc file, determine the width and height of the accompnying binary files.  
        NB GACOS uses top left for the lons and lats origin.  
        """
        f = open(rsc_path, "r")
        lines = f.readlines()
           
        width = int(lines[0].split(' ')[-1][:-1])
        height = int(lines[1].split(' ')[-1][:-1])
        
        lon_topleft = float(lines[6].split(' ')[-1][:-1])
        lat_topleft = float(lines[7].split(' ')[-1][:-1])
        lon_step =  float(lines[8].split(' ')[-1][:-1])
        lat_step =  float(lines[9].split(' ')[-1][:-1])
        
        lons = np.linspace(lon_topleft, lon_topleft + (lon_step * width), width)
        lats = np.linspace(lat_topleft, lat_topleft + (lat_step * height), height)
        
        lons_mg, lats_mg = np.meshgrid(lons, lats)
        
        return width, height, lons_mg, lats_mg
    
    import glob
    
    # 1: get the height and width from a .rsc (resource) file
    rsc_files = glob.glob(str(gacos_dir / '*.rsc'))                                                                              # 
    if len(rsc_files) == 0:
        raise Exception(f"Unable to find any GACOS .rsc files so exiting.  Perhaps the path to the GACOS directory is wrong?     ")
    width, height, lons_mg, lats_mg = width_height_from_gacos_rsc(rsc_files[0])

    # 2: open the gacos binary files and create a data cube of these.  
    gacos_files = sorted(glob.glob(str(gacos_dir / '*.ztd')))                                                                              # 
    n_acq = len(gacos_files)
    acq_dates = []
    gacos_datas = np.zeros((n_acq, height, width) )
    
    #3: Get the acquisition dates for each gacos file.  
    for file_n, gacos_file in enumerate(gacos_files):
        acq_dates.append(gacos_file.split('/')[-1].split('.')[0])                                       # get the date for that gacos atmosphere
        gacos_datas[file_n,] = np.fromfile(gacos_file, dtype=np.float32).reshape((height, width))
        
    
    # 4: if lons and lats are provided (e.g. for each LiCSBAS pixel), resample the GACOS data to that grid.  
    if (lons_mg_new is not None) and (lats_mg_new is not None):
        from scipy.interpolate import griddata
        gacos_datas_resampled = np.zeros((n_acq, lons_mg_new.shape[0], lons_mg_new.shape[1]))                                # initiate a new array that is the size of the resampled (new) data
        gacos_points = np.hstack((np.ravel(lons_mg)[:,np.newaxis], np.ravel(lats_mg)[:,np.newaxis]))                        # the points where we have gacos data, n_points x 2
        print(f"Resamping the GACOS data to the new grid.  Done ", end = '')
        for data_n, gacos_data in enumerate(gacos_datas):
            gacos_datas_resampled[data_n, ] = griddata(gacos_points, np.ravel(gacos_data), (lons_mg_new, lats_mg_new), method='nearest')       # do the resampling for one gacos
            print(f"{data_n} ", end = '')
        return gacos_datas, gacos_datas_resampled, acq_dates, lons_mg, lats_mg
    else:
        return gacos_datas, acq_dates, lons_mg, lats_mg


#%%

from pathlib import Path


def generate_gacos_dates(acq_dates, outdir = Path('.')):
    """ Given a list of acquisition dates, create files with only 20 entries ready to be coped into the GACOS website
    Inputs:
        acq_dates | list of strings | of form YYYYMMDD
        outdir | pathlib Path | directory to write the .txts to
    Returns:
        files
    History:
        2021_09_18 | MEG | Written
    """
    import numpy as np
    
    n_dates = len(acq_dates)
    n_gacos_files = int(np.ceil(n_dates / 20))                                  # gacos only accepts up to 20 dates

    for n_gacos_file in range(n_gacos_files):                                                                   # loop through each file
        with open(outdir / f"gacos_file_{n_gacos_file:03d}.txt", "w") as text_file:
            if n_gacos_file != n_gacos_files:                                       # for all but the last file
                for acq_date in acq_dates[n_gacos_file * 20 : (n_gacos_file+1) * 20 ]:    
                    text_file.write(f"{acq_date}\n")
            else:
                for acq_date in acq_dates[n_gacos_files * 20 :  ]:
                    text_file.write(f"{acq_date}\n")