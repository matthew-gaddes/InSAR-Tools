#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:41:54 2021

@author: matthew

"""


#%%
def licsbas_rationalizer(licsbas_dir, outdir, 
                         licsar_ifgs_ml = {'dir' : 'GEOCml1', 'ifgs' : True, 'aux' :  True},
                         licsar_ifgs_clip = {'dir' : 'GEOCmldirclip', 'ifgs' : True, 'aux' :  True},
                         step_dirs = {'11' : True, '12' : True, '13' : True }):
    """ Given a folder of LiCSBAS outputs, extract only those that are deemed to be interesting/useful.  Primarily useful as the LiCSBAS directory can be huge (difficult to copy), 
    yet retain relatively few useful files.  In addition to the optional files listed below (licsar_ifgs, licsar_ifgs_ml, and steps_dirs), some small files are non-optional (e.g. the h5 file)
    
    Inputs:
        licsbas_dir | pathlib Path | path to LiCSBAS directory
        outdir | pathlib Path | path to rationalized LiCSBAS directory
        licsar_ifgs_ml | dict | sets which of the licsar interferograms to copy.  Usualy the largest directory, and the least useful (unless you want to avoid re-downloading them)
        licsar_ifgs_ml_clip | dict | sets which of the clipped licsar interferograms to copy.  Smaller then previous as clipping has occured.  
                                     
        step_dirs | dict | step 11, 12 and 13 results in the TS_... folder can be copied, if required.  
    
    History:
        2021_07_29 | MEG | Written
        2021_09_21 | MEG | Move from Jasmin to insar_tools


    """
    import os
    import shutil
    import fnmatch



    # 1: Create the ouput directory:
    try:
        os.mkdir(outdir)
    except:
        print(f"Failed to create the output directory.  Would you like to try and delete it and remake it ('y' or 'n')?")
        delete_flag = input()
        if delete_flag == 'y':
            try:
                shutil.rmtree(outdir)
                os.mkdir(outdir)
            except:
                raise Exception(f"Failed to create the out directory ({outdir}) so can't continue.  Exiting.  ")
        elif delete_flag == 'n':
            raise Exception(f"Failed to create the out directory ({outdir}) so can't continue.  Exiting.  ")
        else:
            raise Exception(f"Didn't understand the user input (expected 'y' or 'n') so failed to create the out directory ({outdir}) so can't continue.  Exiting.  ")
                


    # 2: Possibly copy the LiCSAR files (both the originals and the clipped, which use roughly the same directory/naming structure)
    for licsar_ifgs in [licsar_ifgs_ml, licsar_ifgs_clip]:

        if (licsar_ifgs['ifgs'] == True) or (licsar_ifgs['aux']  ==  True):                                                         # only need to make a directory for products if at least one of these is True
            os.mkdir(outdir / licsar_ifgs['dir'])                                                                                   # make he new directory

        if (licsar_ifgs['ifgs'] == True):
            licsar_ifgs_directories = [f.name for f in os.scandir(licsbas_dir / licsar_ifgs['dir']) if f.is_dir()]                                  # get directories of licsar products
            for licsar_ifgs_directory in licsar_ifgs_directories:
                shutil.copytree(licsbas_dir / licsar_ifgs['dir'] / licsar_ifgs_directory, outdir / licsar_ifgs['dir'] / licsar_ifgs_directory)      # copy the directories

        if (licsar_ifgs['aux'] == True):    
            try:
                licsar_ifgs_files = [f.name for f in os.scandir(licsbas_dir / licsar_ifgs['dir']) if not f.is_dir()]                                    # get the licsar files
                for licsar_ifgs_file in licsar_ifgs_files:
                    shutil.copy(licsbas_dir / licsar_ifgs['dir'] / licsar_ifgs_file, outdir / licsar_ifgs['dir'] / licsar_ifgs_file)                     # copy the files
            except:
                print(f"Unable to find and LiCSAR interferograms in {licsar_ifgs['dir']} but trying to continue anyway.   ")
        
            
    # 4: copy the .h5 file(s) and any others in the ts_dir - not currently optional.  
    
    licsbas_dir_dirs = [f.name for f in os.scandir(licsbas_dir ) if f.is_dir()]                                               # get the names of the directories made by LiCSBAS
    for licsbas_dir_dir in licsbas_dir_dirs:                                                                                      # loop through them
        if licsbas_dir_dir[:2] == 'TS':                                                                                        # if it starts with TS, it's the directory we're after      
            ts_dir = licsbas_dir_dir                                                                                          # so assign it to be the name of the ts_dir (time series directory.  )  
    os.mkdir(outdir / ts_dir)                                                                                   # make the new directory
    ts_files = [f.name for f in os.scandir(licsbas_dir / ts_dir) if not f.is_dir()]                             # get the licsar files
    for ts_file in ts_files:
        shutil.copy(licsbas_dir / ts_dir / ts_file, outdir / ts_dir / ts_file)                     # copy the files
        
    for aux_dir in ['info', 'network', 'results']:
        shutil.copytree(licsbas_dir / ts_dir / aux_dir, outdir / ts_dir / aux_dir)      # copy the directories
        
    
    # 3: Possibly copy the step 11/12/13 files:
    for step_n, copy_status in step_dirs.items():
        if copy_status:
            ts_dirs = [f.name for f in os.scandir(licsbas_dir / ts_dir) if f.is_dir()]                                    # get the ts directories
            matches = fnmatch.filter(ts_dirs, f"{step_n}*")                                                                 # get which ones match the step we are looping through
            for match in matches:
                shutil.copytree(licsbas_dir / ts_dir / match, outdir / ts_dir / match)      # copy the directories
                
    # 5: 
    





#%%


def open_fabien_cf_data(file_path, bad_ifgs, crop_coords):
    """ Open Fabien's CF Matlab file in which each ifg has its own coherence mask, drop some bad ifgs (set by the user),
    then find a mask that can be applied ot the whole time series, apply it, and return the ifgs as row vectors and a mask (that can turn a 
    row vector back to an image).  The image is also cropped.  
    Inputs:
    Returns:
    History:
        2021_04_12 | MEG | Created from a sript.  
    """
    
    import datetime as dt
    from insar_tools import daisy_chain_from_acquisitions, baseline_from_names, acquisitions_from_ifg_dates
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    from pathlib import Path
    import scipy.io as sio
    from insar_tools import r3_to_r2
    
    def mean_coherence_for_each_ifg(time_series):
        """ Given a time series of shape (ny, nx, n_ifgs), calculate the ratio of pixels that are not masked for each ifg.  """
        
        mask_each = np.where(time_series == 0, np.ones(time_series.shape), np.zeros(time_series.shape))             # initiate mask
        mean_coh = np.mean(mask_each, axis = (0,1))
        title = 'mean_coherence_in_ifgs'
        f, ax = plt.subplots()    
        ax.plot(np.arange(len(mean_coh)), mean_coh)
        f.suptitle(title)
        f.canvas.set_window_title(title)
    
    
    ICASAR_geocode = {'ge_kmz' : True}                                          # If True, ICASAR will geocode the ICs to view in Google Earth.  
    
    # 0: Open the data and update the dict that contains the geocoding info.  
    mat_contents = sio.loadmat(str(file_path))
    ph_unw_cum = mat_contents['Sinv']                                                                                         #  in metres?  Fabien's email says cumulative.  
    ph_unw_cum = ph_unw_cum[crop_coords['y_start']:crop_coords['y_stop'], crop_coords['x_start']:crop_coords['x_stop'], :]    # crop  
    ph_unw = np.diff(ph_unw_cum, axis = -1)                                                                                   # convert to incremental  
    ny, nx, n_ifgs = ph_unw.shape                           
    lons_r1 = mat_contents['LON'][0,crop_coords['x_start']:crop_coords['x_stop']]                              # crop, used to geocode the outputs
    lats_r1 = mat_contents['LAT'][0,crop_coords['y_start']:crop_coords['y_stop']]                        # crop, used to geocode the outputs, 
    lons_r2 = np.repeat(lons_r1[np.newaxis,:], ny, axis = 0)
    lats_r2 = np.repeat(lats_r1[:, np.newaxis], nx, axis = 1)    
    ICASAR_geocode['lons'] = lons_r2
    ICASAR_geocode['lats'] = lats_r2                                                                        # checked that highest lats are at top of image.  
    
    # 1: deal with the acquisition times etc.  
    # baseline_info = {'baselines' : np.ravel(np.diff(mat_contents['tscene'], axis = 0))}                                    # get the temporal baselines (e.g. 12, 34, 6 etc.  )
    # baseline_info['baselines_cumulative'] = np.cumsum(baseline_info['baselines'])                                          # sum these to get the cumulative temporal baselines.  
    
    acq_dates = []                                                                                    # initiate
    tscenes = mat_contents['tscene']-366                                                              # these are days since 1900, but not sure why Fabien drop 366 from it
    for tscene in tscenes:                                                                            # loop through them
        tscene_dt=dt.datetime.fromordinal(tscene)               
        acq_dates.append(tscene_dt.strftime('%Y%m%d'))
    #baseline_info = {'imdates' : acq_dates}                                                             # add to the dict with all temporal info in.  
    daisy_chain_dates = daisy_chain_from_acquisitions(acq_dates)                                        # same length as ph_unw (as that is incremental)
    del acq_dates
  
    
    # 2: remove any bad ifgs and deal with ifg names and time info etc.  
    #mean_coherence_for_each_ifg(ph_unw)                                 # look for bad ifgs, not needed if we already know where they are (see bad_ifgs argument)

    for bad_ifg in bad_ifgs:
        ph_unw = np.delete(ph_unw, bad_ifg, axis = -1)                                                # incremental, so the daisy chains
        del daisy_chain_dates[bad_ifg]                                                                  # also the daisy chains
    baseline_info = {'ifg_dates' : daisy_chain_dates}
    del daisy_chain_dates
    baseline_info['acq_dates'] = acquisitions_from_ifg_dates(baseline_info['ifg_dates'])
    baseline_info['baselines'] = baseline_from_names(baseline_info['ifg_dates'])
    baseline_info['baselines_cumulative'] = np.cumsum(baseline_info['baselines'])                                          # sum these to get the cumulative temporal baselines.  
    
        
    #mean_coherence_for_each_ifg(ph_unw)                                 # look for bad ifgs again.  
    
    # 3: Calculate a mask for the data
    _,_, n_ifgs = ph_unw.shape                                                                      # upate the number of ifgs after deleting the bad ones.  
    n_pixs = []                                                                                     # initiate to store showing fraction of pixels coherent in each ifg                                                         
    mask = np.where(ph_unw[:,:,0] == 0, np.ones((ny, nx)), np.zeros((ny, nx)))                      # initiate mask
    for n_ifg in np.arange(1, n_ifgs):
        # if np.remainder(n_ifg, 1) == 0:
        #     matrix_show(mask)
        mask_next = np.where(ph_unw[:,:,n_ifg] == 0, np.ones((ny, nx)), np.zeros((ny, nx)))         # where 0, mask is true so make 1.  0 for where mask is not true.  
        mask = np.logical_or(mask, mask_next)
        n_pixs.append(np.mean(mask))                                                                # average numver of coherent pixels is just the mean of the mask
    
    # 4: figure showing results of steps 1 and 2
    f, axes = plt.subplots(1,3)                                                                     # figure to describe Campi Flegrei
    axes[0].plot(np.arange(len(n_pixs)), n_pixs)                                                    # plot the fraction of pixels that are coherent in each ifg.  This probably won't show much if the bad ifgs have been removed!
    axes[0].set_ylim(0,1)
    axes[0].set_title('Fraction of coherent pixels in each ifg')
    axes[1].imshow(mask)                                                                            # the mask taht is applied ot the whole of hte time series
    axes[1].set_title('mask')                                                                       
    cumifg = axes[2].imshow(ma.array(ma.sum(ph_unw, axis = 2), mask = mask))                        # the cumulative ifg for the whole time series.  
    cumifg_cb = f.colorbar(cumifg, ax = axes[2], orientation = 'horizontal')
    axes[2].set_title('Sum of incremental ifgs')
    
    # 5: package data in a form that is ready for LiCSAlert/ICASAR
    ph_unw_r3 = ma.array(ph_unw, mask = np.repeat(mask[:,:,np.newaxis], ph_unw.shape[2], axis = 2))             # rank 3 masked array, using ph_unw as the data and repeating the r2 mask to make it r3
    ph_unw_r3 = np.rollaxis(ph_unw_r3, 2, 0)                                                                    # convert from n_ifgs last dimension to first dimension.  
    displacement_r2 = r3_to_r2(ph_unw_r3)                                                                       # convert to a mask and row vectors (ie. ready for ICA etc.  )
    displacement_r2['incremental'] = displacement_r2.pop('ifgs')                                                # remove and return the value given by the key, which we reassign to the dict.  Essentially changing the name 'ifgs' to 'incremental'

            
    return displacement_r2, baseline_info, ICASAR_geocode

