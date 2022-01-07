#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:35:04 2021

@author: matthew
"""

#%%

def baselines_from_names(names_list):
    """Given a list of ifg names in the form YYYYMMDD_YYYYMMDD, find the temporal baselines in days_elapsed (e.g. 12, 6, 12, 24, 6 etc.  )
    Inputs:
        names_list | list | in form YYYYMMDD_YYYYMMDD
    Returns:
        baselines | list of ints | baselines in days
    History:
        2020/02/16 | MEG | Documented
    """

    from datetime import datetime

    baselines = []
    for file in names_list:

        master = datetime.strptime(file.split('_')[-2], '%Y%m%d')
        slave = datetime.strptime(file.split('_')[-1][:8], '%Y%m%d')
        baselines.append(-1 *(master - slave).days)
    return baselines

#%%

def ifg_dates_dc_to_cum(ifg_dates):
    """Given the names of the daisy chain of interfergorams, find the names of the cumulative ones relative to the first acquisition.  
    Inputs:
        ifg_dates | list | names of daisy chain ifgs, in form YYYYMMDD_YYYYMMDD
    Returns:
        ifg_dates_cum | list | names of cumulative ifgs, in form YYYYMMDD_YYYYMMDD
    History:
        2021_12_09 | MEG | Written
        
    """
    
    ifg_dates_cum = []
    for ifg_date in ifg_dates:
        ifg_dates_cum.append(f"{ifg_dates[0][:8]}_{ifg_date[9:]}")
    return ifg_dates_cum


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

#%% Get the temporal baselines from a folder of SLCs - WIP (needs converting to a function)

# def temporal_baselines_from_SLCs(path):
#     """
#     WIP - never tested.
#     """

#     import glob
#     import os
#     import numpy as np
#     from small_plot_functions import quick_linegraph
#     import datetime

#     # get the files
#     PhUnw_files = sorted(glob.glob(path + '*.zip'), key = os.path.getmtime)            # crude way to sort them so they should be in order

#     # extract the date that each SLC was acquired on
#     dates = []
#     for item in PhUnw_files:
#         dates.append(item[17:25])

#     # sort the dates
#     dates_sorted = sorted(dates, key=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
#     #convert
#     dates_datetimes = []
#     for item in dates_sorted:
#         dates_datetimes.append(datetime.datetime.strptime(item, '%Y%m%d'))


#     temp_baselines = np.zeros((1, len(PhUnw_files) - 1))
#     for i in range(len(PhUnw_files)-1):
#         print(i)
#         temp = dates_datetimes[i+1] - dates_datetimes[i]

#         temp_baselines[0,i] = temp.days
#         del temp
