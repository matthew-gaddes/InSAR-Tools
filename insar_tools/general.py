# -*- coding: utf-8 -*-
"""
"""

#%%



def linear_delay_elveation_correction(dem, incremental_r3, debug_figure = False):
    """Given a time series of ifgs and the dem, correct using a linear relationshiop between height and deformation.  
    Inputs:
        dem | array (or maybe ma)? | Assumes that nans are used for no data, so doesn't need to be a ma
        incremental_r3 | rank 3 masked array | Incremental interferograsms, time first (e.g. 10 x 285, 230 for 10 ifgs)
        debug_figure | boolean | If Ture, figure for each interferogram is produced.  
        
    Returns:
        linear_corrections | rank 3 ma | correction for each interferogram.  SHould be added to ifgs to make correction.  
        incremental_r3_corrected | rank 3 ma | each ifg, after correction has been applied.  
    
    History:
        2022_02_02 | MEG | Written
        
    """
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt

    from scipy.optimize import curve_fit
    
    def linear_fit(x, m, c):
        return m*x + c                                                                       # y = mx + c       
        
    # first deal with masks
    mask_dem = np.isnan(dem)
    mask_ifg = ma.getmask(incremental_r3)[0]
    mask_combined = np.invert(np.logical_and(np.invert(mask_dem), np.invert(mask_ifg)))
    dem_ma = ma.array(dem, mask = mask_combined)
    incremental_r3_combined_mask = ma.array(incremental_r3, mask = np.repeat(mask_combined[np.newaxis,], incremental_r3.shape[0], axis = 0 ))
    
    linear_corrections = np.zeros((incremental_r3_combined_mask.shape))
    incremental_r3_corrected = np.zeros((incremental_r3_combined_mask.shape))
    
    
    for ifg_n, ifg_inc in enumerate(incremental_r3_combined_mask):
        
        params = curve_fit(linear_fit, ma.compressed(dem_ma), ma.compressed(ifg_inc))          # fit line (defiend above) through time series for that point.  
        m = params[0][0]
        c = params[0][1]
        linear_correction = (-1) * ((m * dem_ma) + c)
        
        ifg_inc_corrected = ifg_inc + linear_correction
        
        linear_corrections[ifg_n, ] = linear_correction
        incremental_r3_corrected[ifg_n,] = ifg_inc_corrected
        
        
        if debug_figure:
            f, axes = plt.subplots(2,3)
            axes[0,0].scatter(ma.compressed(dem_ma), ma.compressed(ifg_inc))
            axes[0,1].scatter(ma.compressed(dem_ma), ma.compressed(linear_correction))
            axes[0,2].scatter(ma.compressed(dem_ma), ma.compressed(ifg_inc_corrected))
            vmin = ma.min(ma.concatenate((ma.ravel(ifg_inc), ma.ravel(linear_correction), ma.ravel(ifg_inc_corrected))))
            vmax = ma.max(ma.concatenate((ma.ravel(ifg_inc), ma.ravel(linear_correction), ma.ravel(ifg_inc_corrected))))
            axes[1,0].imshow(ifg_inc, vmin = vmin, vmax = vmax)
            axes[1,1].imshow(linear_correction, vmin = vmin, vmax = vmax)
            axes[1,2].imshow(ifg_inc_corrected, vmin = vmin, vmax = vmax)
            
            title = ['Interferogram', 'Linear correction', 'Interferogram + linear_correction']
            for ax_n, ax in enumerate(axes[0,]):
                ax.set_title(title[ax_n])
                ax.axhline(0)
        
    linear_corrections = ma.array(linear_corrections, mask = ma.getmask(incremental_r3_combined_mask))
    incremental_r3_corrected = ma.array(incremental_r3_corrected, mask = ma.getmask(incremental_r3_combined_mask))
            
    return linear_corrections, incremental_r3_corrected

#%%

def cum_to_vel(cumulative_r3, baselines_cs, fix_0 = True):
    """ Given a rank 3 of cumulative displacements, solve for the velocity (a rank 2),
    and the smoothed cumulative displacements (i.e. the velocity * temporal baseline for each ifg.  Assumes constant velocity through time!
    Inputs:
        cumulative_r3 | rank 3 masked array | cumulative displacments, for each acquisition.  All 0 on first entry.  
        baselines_cs | rank 1 | cuulative temporal baselines, in days.  e.g. 0,6,12,18, 24 if acquisition every 6 days.  
        fix_0 | boolean | If True, lines that is fit through time series of each point must pass through 0 at start.  
                          If false, y intercept ('c') is also solved for.  
    Returns:
        vel_r2 | rank 2 masked array | velocity for each pixel
        cumulative_r3_smooth | rank 3 masked array | cumulative displacement for each pixel, assuming constant velocity in time.  
    History:
        2022_02_02 | MEG | Written.  
       """
    from scipy.optimize import curve_fit
    import numpy as np
    import numpy.ma as ma
    
    if fix_0:
        def fit_func(x, m):
            return m*x + 0                                                                       # y = mx
    else:
        def fit_func(x, m, c):
            return m*x + c                                                                       # or y = mx + c       
        
    
    n_acq, ny, nx = cumulative_r3.shape
    vel_r2  = np.zeros((ny, nx))                                                                # initiate to store velocities in 
    cumulative_r3_smooth = np.zeros((n_acq, ny, nx))                                            # initiate to store smoothed displacments in
    print(f"Starting to compute the velocity for each pixel: Completed row ", end = '')
    for row_n in range(ny):
        for col_n in range(nx):
            params = curve_fit(fit_func, baselines_cs, cumulative_r3[:, row_n, col_n])          # fit line (defiend above) through time series for that point.  
            if fix_0:
                m = params[0]
                cumulative_r3_smooth[:, row_n, col_n] = (m * baselines_cs)                      # predict displacements assuming velocity is always m
            else:
                m = params[0][0]
                c = params[0][1]
                cumulative_r3_smooth[:, row_n, col_n] = (m * baselines_cs) + c                  # predict displacements, assuing velocity is always m and disp at start of time series could be c
            vel_r2[row_n, col_n] = m                                                            # velocity is just m
        print(f"{row_n} ", end = '')
        
    vel_r2 = ma.array(vel_r2, mask = ma.getmask(cumulative_r3[0,]))                             # mask water/incoherence.  
    cumulative_r3_smooth = ma.array(cumulative_r3_smooth, mask = ma.getmask(cumulative_r3))     # mask water/incoherence.  
    return vel_r2, cumulative_r3_smooth

#%%

def reference_r3_ts(ts_r3, ref_x, ref_y):
    """ Given a rank 3 time series (i.e. n_images x ny x nx) and a reference pixel (the same through time), reference the time series to that.  
    Inputs:
        ts_r3 | rank 3 numpy array | images/interfergrams.  
        ref_x | int |  x pixel of reference region.  
        ref_y | int | y pixel of reference region.  
    Returns:
        ts_r3_referenced | rank 3 array.  
    History:
        2021_08_25 | MEG | Written
    """
    import numpy as np
    
    pixel_values_r1 = ts_r3[:, ref_y, ref_x][:,np.newaxis, np.newaxis]                                   # get the value for the reference pixel at all times 
    pixel_values_r3 = pixel_values_r1.repeat(ts_r3.shape[1], 1).repeat(ts_r3.shape[2], 2)                # repeat it so it's the same size as the original time series
    ts_r3_referenced = ts_r3 - pixel_values_r3                                                           # and correct it
    return ts_r3_referenced





#%%


def r2_to_r3(ifgs_r2, mask):
    """ Given a rank2 of ifgs as row vectors, convert it to a rank3. 
    Inputs:
        ifgs_r2 | rank 2 array | ifgs as row vectors 
        mask | rank 2 array | to convert a row vector ifg into a rank 2 masked array        
    returns:
        phUnw | rank 3 array | n_ifgs x height x width
    History:
        2020/06/10 | MEG  | Written
    """
    import numpy as np
    import numpy.ma as ma
    
    n_ifgs = ifgs_r2.shape[0]
    ny, nx = col_to_ma(ifgs_r2[0,], mask).shape                                   # determine the size of an ifg when it is converter from being a row vector
    
    ifgs_r3 = np.zeros((n_ifgs, ny, nx))                                                # initate to store new ifgs
    for ifg_n, ifg_row in enumerate(ifgs_r2):                                           # loop through all ifgs
        ifgs_r3[ifg_n,] = col_to_ma(ifg_row, mask)                                  
    
    mask_r3 = np.repeat(mask[np.newaxis,], n_ifgs, axis = 0)                            # expand the mask from r2 to r3
    ifgs_r3_ma = ma.array(ifgs_r3, mask = mask_r3)                                      # and make a masked array    
    return ifgs_r3_ma



def r3_to_r2(phUnw):
    """ Given a rank3 of ifgs, convert it to rank2 and a mask.  Works with either masked arrays or just arrays.  
    Inputs:
        phUnw | rank 3 array | n_ifgs x height x width
    returns:
        r2_data['ifgs'] | rank 2 array | ifgs as row vectors
        r2_data['mask'] | rank 2 array 
    History:
        2020/06/09 | MEG  | Written
    """
    import numpy as np
    import numpy.ma as ma
    
    if ma.isMaskedArray(phUnw):
        n_pixels = len(ma.compressed(phUnw[0,]))                                            # if it's a masked array, get the number of non-masked pixels
        mask = ma.getmask(phUnw)[0,]                                                        # get the mask, which is assumed to be constant through time
    else:
        n_pixels = len(np.ravel(phUnw[0,]))                                                 # or if a normal numpy array, just get the number of pixels
        mask = np.zeros(phUnw[0,].shape)                                                    # or make a blank mask
 
    r2_ifgs = np.zeros((phUnw.shape[0], n_pixels))                                          # initiate to store ifgs as rows in
    for ifg_n, ifg in enumerate(phUnw):
        if ma.isMaskedArray(phUnw):
            r2_ifgs[ifg_n,] = ma.compressed(ifg)                                            # non masked pixels into row vectors
        else:
            r2_ifgs[ifg_n,] = np.ravel(ifg)                                                 # or all just pixles into row vectors

    r2_data = {'ifgs' : r2_ifgs,                                                            # make into a dictionary.  
               'mask' : mask}          
    return r2_data



#%%
def mask_nans(phUnw_r3, figures = False, threshold = 100):
    """ Given ifgs as a rank3 array that might contain nans, remove any pixels that are nans at any point in the time series.  
    Note that the same mask is applied through time, so if a pixel is nan in a single ifg, it will be removed from the entire 
    set.      
    Inputs:
        phUnw_r3 | r3 array | n_ifgs x height x width        
        figures | boolean | If True, figure will be produces
        threshold | float in range [0 100] | Ifgs with more nans (as a percentrage) above this threshold will be dropped.  
                                             E.g. if 0, all will be dropped, and if 100 none
    Returns:
        phUnw_r3_masked | r3 masked array | n_ifgs x height x width        
    History:
        2020/06/09 | MEG | Written
        2020/06/11 | MEG | Update to include figure of how many pixels are being masked per ifg.  
    """
    
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
       
    # find where the nans are in the ifgs
    n_ifgs = phUnw_r3.shape[0]
    n_pixels = phUnw_r3.shape[1] * phUnw_r3.shape[2]
    nan_args = np.argwhere(np.isnan(phUnw_r3))                                                      # find nans in set of ifgs
    
    # find the number per interferogram
    ifgs_with_nans = np.unique(nan_args[:,0])                                                       # first column is ifg n, so just get each one once.  
    n_nan = np.zeros((n_ifgs, 1))                                                                   # initiate as a column of 0s
    for ifg_with_nan in ifgs_with_nans:
        n_nan[ifg_with_nan] = len(np.argwhere(nan_args[:,0] == ifg_with_nan))
    n_nan_percent = 100 * (n_nan / n_pixels)                                                        # and as a % of the total number 
    
    if figures:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Number of Nans in each interferogram')
        ax.bar(np.arange(n_ifgs), np.ravel(n_nan))
        ax2 = ax.twinx()                                                                         # instantiate a second axes that shares the same x-axis
        ax2.bar(np.arange(n_ifgs), np.ravel(n_nan_percent))
        ax2.axhline(threshold, c = 'r')
        ax.set_xlabel('Ifg. number')
        ax.set_ylabel('Number of nans')
        ax2.set_ylabel('% of total pixels')  # we already handled the x-label with ax1
    
    # drop ifgs that have too many nans
    good_ifg = np.argwhere(n_nan_percent < threshold)    
    print(f"{n_ifgs - len(good_ifg)} interferograms will be dropped for having more nans than the threshold ({threshold} %).  ")
    phUnw_r3 = phUnw_r3[good_ifg[:,0],]
    n_ifgs = phUnw_r3.shape[0]                                                                      # upate the number of interferograms
    nan_args = np.argwhere(np.isnan(phUnw_r3))                                                      # find nans in set of ifgs
    mask_nans = np.zeros(phUnw_r3.shape[1:])                                                        # mask is just zeros to begin with
    for nan_arg in nan_args:                                                                        # loop through each nan point
        mask_nans[nan_arg[1], nan_arg[2]] = 1                                                       # and modify the y and x location of the mask to be 1 if that pixel should be masked
    mask_nans = np.repeat(mask_nans[np.newaxis,], n_ifgs, axis = 0)                                 # increase the size of the mask to be rank 3 (like phUnw)
    phUnw_r3_masked = ma.array(phUnw_r3, mask = mask_nans)                                          # and apply it to phUnw, making it a masked array
    

    return phUnw_r3_masked, n_nan





def set_reference_region(ifgs_r3, ref_region, ref_halfwidth = 20):
    """
    A function to set a time series of interfergrams relative to a reference region
    (i.e. mean of that area/pixel is 0 for each ifg), should be outside deforming and
    masked regions - no check on this!

    Inputs:
        ifgs_r3 | rank 3 masked array | ifgs x heightx width
        ref_region | tuple | xy of pixel in centre of region to be reference
        ref_halfwidrth | int | half of length of sie of square ref region area
    Outputs
        ifgs_r3 | as above
    """
    import numpy as np
    import numpy.ma as ma

    n_ifgs, n_pixs_y, n_pixs_x = ifgs_r3.shape

    ref_means = ma.mean(ifgs_r3[:,ref_region[1]-ref_halfwidth:ref_region[1]+ref_halfwidth,
                                ref_region[0]-ref_halfwidth:ref_region[0]+ref_halfwidth], axis = (1,2))     # calculate mean of refence region for each ifg

    ref_means = ref_means[:,np.newaxis, np.newaxis]                             # expand to rank 3
    ref_means = np.repeat(ref_means, n_pixs_y, axis = 1)                        # continued - y direction
    ref_means = np.repeat(ref_means, n_pixs_x, axis = 2)                        # continued - x direction

    ifgs_r3 -= (ref_means)                                                      # do mean centering

    return ifgs_r3


#%%

def apply_mask_to_r3_ifgs(r3_ifgs, mask_coh):
    """
    Add a new mask to an existing masked array of rank 3 ifgs (n_ifgs x height x width)
    """
    import numpy as np
    import numpy.ma as ma

    mask_coh_r3 = np.repeat(mask_coh[np.newaxis,:,:],r3_ifgs.shape[0], axis = 0 )
    r3_ifgs_mask = ma.array(r3_ifgs, mask = mask_coh_r3)
    return r3_ifgs_mask

#%%

def invert_to_DC(ifg_acq_numbers, tcs):
    """
    Given a list of which acquisitions the time series of interferograms were made,
    returns the time courses for the simplest daisy chain of interferograms.
    From the method described in Lundgren ea 2001
    """
    import numpy as np
    n_ifgs = len(ifg_acq_numbers)
    n_times = np.max(ifg_acq_numbers)                 # this will be the number of model parameters
    d = tcs
    g = np.zeros((n_ifgs, n_times))                 # initiate as zeros

    for i, ifg_acq_number in enumerate(ifg_acq_numbers):
        g[i, ifg_acq_number[0]:ifg_acq_number[1]] = 1
    m = np.linalg.inv(g.T @ g) @ g.T @ d                       # m (n_sources x 1)

    return m





#%% delete all but daisy chain ifgs

def get_daisy_chain(phUnw, ifg_names):
    """Given a an array of LicSAR ifgs and their names (which are slaveData_masterDate),
    return only the daisy chain of ifgs.
    Inputs:
        phUnw | r2 or 3 array | ifgs as (samples x height x width) or as (samples x n_pixels)
        ifg_names | list of strings | of form 20180101_20190102 (for a 1 day ifg in January)
        
    Returns:
        phUnw | r2 or 3 array | ifgs as (samples x height x width) or as (samples x n_pixels), but only daisy chain
        ifg_names | list of strings | of form 20180101_20190102 (for a 1 day ifg in January), but only daisy chain
    
    History:
        2019/??/?? | MEG | Written
        2020/06/02 | MEG | Update to handle phUnw as either rank 2 or rank 3
        
    """
    from insar_tools import files_to_daisy_chain

    _, daisy_chain_ifg_names, _ = files_to_daisy_chain(ifg_names, figures = False)                   # get the names of the daisy chain ifgs

    daisy_chain_ifg_args = []                                                                       # list to store which number ifg each daisy chain ifg is
    for daisy_chain_ifg_name in daisy_chain_ifg_names:                                              # loop through populating hte list
        try:
            daisy_chain_ifg_args.append(ifg_names.index(daisy_chain_ifg_name))
        except:
            print(f'{daisy_chain_ifg_name} could not be found in the interferogram list.  Skipping it.  ')
            pass
    phUnw = phUnw[daisy_chain_ifg_args,]                                                         # select the desired ifgs
    ifg_names = [ifg_names[i] for i in daisy_chain_ifg_args]                                        # select the ifg names

    return phUnw, ifg_names


#%%




#%%

def mask_ifgs(ph, mask_old, mask_new):
    """
    Take some masked ifgs and apply a new mask to them (that is bigger than the original one)
    Inputs:
        ph | r2 array | ifgs as row vectors
        mask_old | r2 boolean | old mask
        mask_new | r2 boolean | new mask
    """
    import numpy as np
    import numpy.ma as ma

    for i, row in enumerate(ph):
        temp = col_to_ma(ph[i,:], mask_old)                                       # convert column to masked array, and mask out the water
        temp2 = ma.filled(temp, fill_value = 1)                                                      # convert masked array to array
        temp3 = ma.array(temp2, mask = mask_new)                                     # and array back to masked array with new amask
        if i == 0:
            ph2 = np.zeros((np.size(ph, axis = 0), len(ma.compressed(temp3))))
        ph2[i,:] = ma.compressed(temp3)
    return ph2

#%% Make a zipfile from a folder of SLCs



def chronological_zipfile_list(path):
    """
    A function to creat a zipfile.list (input for LiCSAR) that has the files in chronlogical order

    A bit of a mess of a script that handles paths badly and will probably need tweaing
    """

    import glob
    import os
    import numpy as np
    #from small_plot_functions import *
    import datetime

    # get all the zipfiles in the directory
    PhUnw_files = sorted(glob.glob('./*zip'), key = os.path.getmtime)            # crude way to sort them so they should be in order

    # extract the date that each SLC was acquired on
    dates = []
    for item in PhUnw_files:
        dates.append(item.split('_')[5][:8])

    # sort the dates
    dates_sorted = sorted(dates, key=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    dates_sorted_args = sorted(range(len(dates)), key=lambda x: datetime.datetime.strptime(dates[x], '%Y%m%d'))

    # write the output file
    file = open('zipfile.list', 'w')
    for i in dates_sorted_args:
        file.write( path + PhUnw_files[i][2:] + '\n')
    file.close()



    quick_linegraph(temp_baselines)
    
#%%

def col_to_ma(col, pixel_mask):
    """ A function to take a column vector and a 2d pixel mask and reshape the column into a masked array.
    Useful when converting between vectors used by BSS methods results that are to be plotted

    Inputs:
        col | rank 1 array |
        pixel_mask | array mask (rank 2)

    Outputs:
        source | rank 2 masked array | colun as a masked 2d array

    2017/10/04 | collected from various functions and placed here.

    """
    import numpy.ma as ma
    import numpy as np

    source = ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask )
    source.unshare_mask()
    source[~source.mask] = col.ravel()
    return source

