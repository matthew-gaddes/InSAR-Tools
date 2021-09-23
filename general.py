# -*- coding: utf-8 -*-
"""
 A selection of functions for helping to work with LiCSAR

Common paths:
path = '/nfs/a1/homes/eemeg/galapagos_from_asf_v2/'                 # path to the SLC files
path = '/nfs/a1/homes/eemeg/data_etna_from_asf/'                 # path to the SLC files

"""


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
    from small_plot_functions import col_to_ma
    
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





#%%

def daisy_chain_from_acquisitions(acquisitions):
    """Given a list of acquisiton dates, form the names of the interferograms that would create a simple daisy chain of ifgs.
    Inputs:
        acquisitions | list | list of acquistiion dates in form YYYYMMDD
    Returns:
        daisy_chain | list | names of daisy chain ifgs, in form YYYYMMDD_YYYYMMDD
    History:
        2020/02/16 | MEG | Written
    """
    daisy_chain = []
    n_acqs = len(acquisitions)
    for i in range(n_acqs-1):
        daisy_chain.append(f"{acquisitions[i]}_{acquisitions[i+1]}")
    return daisy_chain

#%% 

def acquisitions_from_ifg_dates(ifg_dates):
    """Given a list of ifg dates in the form YYYYMMDD_YYYYMMDD, get the uniqu YYYYMMDDs that acquisitions were made on.  
    Inputs:
        ifg_dates | list | of strings in form YYYYMMDD_YYYYMMDD.  Called imdates in LiCSBAS nomenclature.  
    Returns:
        acq_dates | list | of strings in form YYYYMMDD
    History:
        2021_04_12 | MEG | Written
        
    """
    
    acq_dates = []
    for ifg_date in ifg_dates:                                          # loop through the dates for each ifg
        dates = ifg_date.split('_')                                     # split into two YYYYMMDDs
        for date in dates:                                              # loop through each of these
            if date not in acq_dates:                                   # if it't not already in the list...
                acq_dates.append(date)                                  # add to it
    return acq_dates


#%%


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




#%%

def files_to_daisy_chain(ifg_names, figures = True):
    """
    Given a list of interfergram names (masterDate_slaveDate), it:
        - finds all the acquisition dates
        - forms a list of names of the simplest daisy chain of interfegrams we can make
        - lists which number acquistion each interferogram is between (e.g (0, 3))
    Inputs:
        ifg_names | list of strings | of form 20180101_20190102 (for a 1 day ifg in January)
        figures | boolean | For the figure output of the termporal baselines.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta

    # get acquistion dates (ie when each SAR image was taken)
    dates_acq = []
    for date in ifg_names:
        date1 = date[:8]
        date2 = date[9::]
        if date1 not in dates_acq:
            dates_acq.append(date1)
        if date2 not in dates_acq:
            dates_acq.append(date2)
    dates_acq = sorted(dates_acq)

    # get the dates for the daisy chain of interferograms (ie the simplest span to link them all)
    daisy_chain_ifgs= []
    for i in range(len(dates_acq)-1):
        daisy_chain_ifgs.append(dates_acq[i] + '_' + dates_acq[i+1])

    # get acquestion dates in terms of days since first one
    days_elapsed = np.zeros((len(dates_acq), 1))
    first_acq = datetime.strptime(ifg_names[0][:8], '%Y%m%d')
    for i, date in enumerate(dates_acq):
        date_time = datetime.strptime(date, '%Y%m%d')
        days_elapsed[i,0] = (date_time - first_acq).days

    # find which acquisiton number each ifg spans
    ifg_acq_numbers = []
    for i, ifg in enumerate(ifg_names):
        master = ifg[:8]
        slave = ifg[9:]
        pair = (dates_acq.index(master), dates_acq.index(slave))
        ifg_acq_numbers.append(pair)



    if figures:                                                             # temp baseline plot
        f, ax = plt.subplots(1)
        for i, file in enumerate(ifg_names):
            master = datetime.strptime(file[:8], '%Y%m%d')
            slave = datetime.strptime(file[9:], '%Y%m%d')
            master_xval = (master - first_acq).days
            slave_xval = (slave - first_acq).days
            ax.plot((master_xval, slave_xval), (i,i), '-')
        for i in range(len(dates_acq)-1):
            master = datetime.strptime(dates_acq[i], '%Y%m%d')
            slave = datetime.strptime(dates_acq[i+1], '%Y%m%d')
            master_xval = (master - first_acq).days
            slave_xval = (slave - first_acq).days
            ax.plot((master_xval, slave_xval), (-len(dates_acq)+i,-len(dates_acq)+i), '-', c = 'k')
        ax.set_ylabel('Ifg. #')
        ax.set_xlabel('Days since first acquisition')

    return dates_acq, daisy_chain_ifgs, ifg_acq_numbers

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

def baseline_from_names(names_list):
    """Given a list of ifg names in the form YYYYMMDD_YYYYMMDD, find the temporal baselines in days_elapsed (e.g. 12, 6, 12, 24, 6 etc.  )
    Inputs:
        names_list | list | in form YYYYMMDD_YYYYMMDD
    Returns:
        baselines | list of ints | baselines in days
    History:
        2020/02/16 | MEG | Documented
    """

    from datetime import datetime, timedelta

    baselines = []
    for file in names_list:

        master = datetime.strptime(file.split('_')[-2], '%Y%m%d')
        slave = datetime.strptime(file.split('_')[-1][:8], '%Y%m%d')
        baselines.append(-1 *(master - slave).days)
    return baselines


#%%

def mask_ifgs(ph, mask_old, mask_new):
    """
    Take some masked ifgs and apply a new mask to them (that is bigger than the original one)
    Inputs:
        ph | r2 array | ifgs as row vectors
        mask_old | r2 boolean | old mask
        mask_new | r2 boolean | new mask
    """
    from small_plot_functions import col_to_ma
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


#%% Get the temporal baselines from a folder of SLCs - WIP (needs converting to a function)

def temporal_baselines_from_SLCs(path):
    """
    WIP - never tested.
    """

    import glob
    import os
    import numpy as np
    from small_plot_functions import quick_linegraph
    import datetime

    # get the files
    PhUnw_files = sorted(glob.glob(path + '*.zip'), key = os.path.getmtime)            # crude way to sort them so they should be in order

    # extract the date that each SLC was acquired on
    dates = []
    for item in PhUnw_files:
        dates.append(item[17:25])

    # sort the dates
    dates_sorted = sorted(dates, key=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    #convert
    dates_datetimes = []
    for item in dates_sorted:
        dates_datetimes.append(datetime.datetime.strptime(item, '%Y%m%d'))


    temp_baselines = np.zeros((1, len(PhUnw_files) - 1))
    for i in range(len(PhUnw_files)-1):
        print(i)
        temp = dates_datetimes[i+1] - dates_datetimes[i]

        temp_baselines[0,i] = temp.days
        del temp


    quick_linegraph(temp_baselines)
