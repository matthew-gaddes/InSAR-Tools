# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:02:02 2017

A collection of functions for plotting synthetic interferograms

@author: eemeg
"""


#%%


def xticks_every_nmonths(ax_to_update, day0_date, time_values, include_tick_labels, 
                         major_ticks_n_months = 3, minor_ticks_n_months = 1):
    """Given an axes, update the xticks so the major ones are the 1st of every n months (e.g. if every 3, would be: jan/april/july/october).  
    
    Inputs:
        ax_to_update | matplotlib axes | the axes to update.  
        day0_date | string | in form yyyymmdd
        time_values | rank 1 array | cumulative temporal baselines, e.g. np.array([6,18, 30, 36, 48])
        include_tick_labels | boolean | if True, tick labels are added to the ticks.  
        n_months | int | x ticks are very n months.  e.g. 2, 3,4,6,12 (yearly)  Funny spacings (e.g. every 5) not tested.  
    Returns:
        updates axes
    History:
        2021_09_27 | MEG | Written
        2022_02_17 | MEG | modify so can be monhtly spacing other than every 3 months.  
    """
    import pdb
    import numpy as np
    import datetime as dt
    import copy
    
    from dateutil.relativedelta import relativedelta                                                    # add 3 months and check not after end
    import matplotlib.pyplot as plt

    
    def create_tick_labels_every_nmonths(day0_date_dt, dayend_date_dt, n_months = 1):
        """ Given a spacing of every n_months, get the dates and days since the first date for ticks every n_months.  
        e.g. every month, every 6 months.  
        
        Inputs:
            day0_date_dt | datetime | date of x = 0 on axis.  
            dayend_date_dt | datetime | date of last x value. 
            n_months | int | frequency of ticks. 
            
        Returns:
            ticks | dict | contains datetimes : datetimes for each tick
                                    yyyymmdd : strings to use as labels in form yyyy/mm/dd
                                    n_day     : day number of tick.  
        History:
            2022_03_29 | MEG | Written
        """
    
        # 1: find first tick date (the first of the jan/ april/jul /oct)                        
        date_tick0 = copy.deepcopy(day0_date_dt)                                                                 # version that can be modified as we iterate through.  
        while not ( (date_tick0.day) == 1 and (date_tick0.month in (np.arange(0, 12, n_months) + 1))):           # i.e. whilst it's not the 1st of the month, and not jan/apr/jul/oct....
            date_tick0 +=  dt.timedelta(1)                                                                       # then add one day and keep going.  
    
        # 2: get all the other first of the quarters as datetimes (ie keep adding n months until we're gone past the day end date)
        ticks = {'datetimes' : [date_tick0],
                 'yyyymmdd'   : [],
                 'n_day'     : []}
       
        while ticks['datetimes'][-1] < (dayend_date_dt - relativedelta(months=+n_months)):                         # while we haven't gone past the last date (and subtract 3 months to make sure we don't go one 3 month jump too far. )
            ticks['datetimes'].append(ticks['datetimes'][-1] + relativedelta(months=+n_months))                    # append the next date which is n_months more.  
        
        # 3: work out what day number each first of the quarter is.  
        for tick_dt in ticks['datetimes']:                                                                      # loop along the list of datetimes (which are each tick) 
            ticks['yyyymmdd'].append(dt.datetime.strftime(tick_dt, "%Y/%m/%d"))                                 # as a string that can be used for the tick label (hence why include / to make more readable)
            ticks['n_day'].append((tick_dt - day0_date_dt).days)
            
        return ticks

    
    xtick_label_angle = 315                                                                              # this angle will read from top left to bottom right (i.e. at a diagonal)
    
    tick_labels_days = ax_to_update.get_xticks().tolist()                                                # get the current tick labels
    day0_date_dt = dt.datetime.strptime(day0_date, "%Y%m%d")                                             # convert the day0 date (date of day number 0) to a datetime.  
    dayend_date_dt = day0_date_dt +  dt.timedelta(int(time_values[-1]))                                  # the last time value is the number of days we have, so add this to day0 to get the end.  

    ticks_major = create_tick_labels_every_nmonths(day0_date_dt, dayend_date_dt, n_months = major_ticks_n_months)
    ticks_minor = create_tick_labels_every_nmonths(day0_date_dt, dayend_date_dt, n_months = minor_ticks_n_months)                        # these are used as the minor ticks every month.  
    

        
    # 4: Update the figure.  
    ax_to_update.set_xticks(ticks_major['n_day'])                                                                   # apply major tick labels to the figure
    ax_to_update.set_xticks(ticks_minor['n_day'], minor = True)                                                                   # apply major tick labels to the figure

    if include_tick_labels:
        ax_to_update.set_xticklabels(ticks_major['yyyymmdd'], rotation = xtick_label_angle, ha = 'left')            # update tick labels, and rotate
        plt.subplots_adjust(bottom=0.15)
        ax_to_update.set_xlabel('Date')
    else:
        ax_to_update.set_xticklabels([])                                                                    # remove any tick lables if they aren't to be used.  
    
    # add vertical lines every year.  
    for major_tick_n, datetime_majortick in enumerate(ticks_major['datetimes']):
        if datetime_majortick.month == 1:                                                                       # if it's the january tick (i.e the 1st of the year)
            ax_to_update.axvline(x = ticks_major['n_day'][major_tick_n], color='k', alpha=0.1, linestyle='--')                          
               


#%%


def plot_points_interest(r3_data, points_interest, baselines_cs, acq_dates, title = '', ylabel = 'm',
                         ylims = None):
    """ Given rank 3 data of incremental interferograms (e.g. n_images x height x width) and some points of interest (e.g. an xy pair), plot the cumulative time
    series for those points (i.e. as r3 is incremental, summing is done in the function).  Also information is required (baselines and acq_dates) for the x axis of the plot.  
    
    Inputs:
        r3_data | rank 3 array (masked array support?) | incremental interferograms, rank 3 (n_images x height x width)
        points_interest | dict | point name (e.g. 'reference' or 'deforming' and tuple of x and y position.  )
        baselines_cs | rank 1 array | cumulative temporal baselines in days.  
        acq_dates | string of YYYYMMDD | date of each radar acquisition.  
        title | string | figure and window title.  
        ylabel | string | units of r3_data (e.g. m, mm, cm, rad etc.  )
    Returns:
        Figure
    History:
        2021_09_22 | MEG | Added to package.  
    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    from scipy import optimize
    
    def test_func(x, a, b, c):
        """ a sets amplitude, b sets frequency, c sets gradient of linear term
        """
        return c * x + (a * np.sin((2*np.pi *(1/b) * x)))
    

    
    f, ax = plt.subplots(figsize = (10,6))
    f.canvas.manager.set_window_title(title)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(ylabel)
    ax.axhline(0,c = 'k')
    for key, value in points_interest.items():
        ax.scatter(baselines_cs, np.cumsum(r3_data[:,value[1], value[0]]), label = key)              # plot each of hte points.  
    try:
        params, params_covariance = optimize.curve_fit(test_func, baselines_cs, np.cumsum(r3_data[:,points_interest['highlands'][1], 
                                                                                                    points_interest['highlands'][0]]), p0=[15, 365, 0.01])            # p0 is first guess at abc parameters for sinusoid (ie. 365 means suggesting it has an annual period)
        y_highlands_predict = test_func(baselines_cs, params[0], params[1], params[2])                                  # predict points of line.  
        ax.plot(baselines_cs, y_highlands_predict, c='k', label = 'Sinusoid + linear')                          # plot the line of best fit.  
    except:
        print(f"Failed to find a highlands point to fit a ling of best fit to, but continuing anyway.  ")
    ax.legend()
    
    if ylims is not None:
        ax.set_ylim(bottom = ylims['bottom'], top = ylims['top'])
    
    
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 180))
    xticks_dayn = ax.get_xticks()
    xticks_date = []
    day0_date = datetime.strptime(acq_dates[0], '%Y%m%d')
    for xtick_dayn in xticks_dayn:
        xtick_date = day0_date + timedelta(days = float(xtick_dayn))                  # add the number of dats to the original date
        xticks_date.append(xtick_date.strftime('%Y_%m_%d'))
    ax.set_xticklabels(xticks_date, rotation = 'vertical')
    f.subplots_adjust(bottom=0.2)



#%%


    

def BssResultComparison(S_synth, tc_synth, S_pca, tc_pca, S_ica, tc_ica, pixel_mask, title):
    """ A function to plot the results of PCA and ICA against the synthesised sources
    Inputs:
        S_synth | rank 2 array | synthesised sources images as rows (e.g. 2 x 5886)
        tc_synth | rank 2 array | synthesised time courses as columns (e.g. 19 x 2)
        S_pca | PCA sources
        tc_pca | PCA time courses
        S_ica | ica sources
        tc_ica | ica time courses
        pixel_mask | rank 2 pixel mask | used to covnert columns arrays to 2d masked arrays (ie for ifgs)
        title | string | figure name
        
    """
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.gridspec as gridspec 
    
      
    def source_row(sources, tcs, grid):
        """ Given a grid object, plot up to 6 spatial maps and time courses in a 2x6 grid
        """
        def linegraph(sig, ax):
            """ signal is a 1xt row vector """
            times = sig.size
            if times > 20:
                times = 20
            a = np.arange(times)
            ax.plot(a,sig[:20], color='k')
            ax.axhline(y=0, color='k', alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            #ax.set_aspect(1)
        def ifg(sig, ax):
            """ signal is a 1xt row vector """
            from small_plot_functions import col_to_ma
            #ax.imshow(col_to_ma(sig, pixel_mask), cmap = matplotlib.cm.coolwarm, vmin = -1, vmax = 1)
            ax.imshow(col_to_ma(sig, pixel_mask), cmap = matplotlib.cm.coolwarm)
            ax.set_xticks([])
            ax.set_yticks([])
            
        grid_inner = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=grid, wspace=0.0, hspace=0.0)
        for j in np.arange(0,6):
            if j < np.size(sources, axis= 0):
                ax_ifg = plt.subplot(grid_inner[0, j])
                ax_line = plt.subplot(grid_inner[1, j])
                ifg(sources[j,:], ax_ifg)
                linegraph(tcs[:,j], ax_line)
        

    fig_extraSources_comps = plt.figure(title, figsize=(8,8))
    grid_rows = gridspec.GridSpec(3, 1)
    source_row(S_synth, tc_synth, grid_rows[0])
    source_row(S_pca, tc_pca, grid_rows[1])
    source_row(S_ica, tc_ica, grid_rows[2])
        
    fig_extraSources_comps.tight_layout(rect =[0.05,0,1,1])
    fig_extraSources_comps.text(0.05, 0.88, 'Sources', fontsize=12, rotation = 90, horizontalalignment='center')
    fig_extraSources_comps.text(0.05, 0.55, 'sPCA', fontsize=12, rotation = 90, horizontalalignment='center')
    fig_extraSources_comps.text(0.05, 0.24, 'sICA', fontsize=12, rotation = 90, horizontalalignment='center')

#%% ifg plot



def ifg_plot_V2(ifgs, pixel_mask, cols, title, colorbar=False, shared=True):
    """
    Function to plot a time series of ifg    
    ifgs | pxt matrix of ifgs as columns (p pixels, t times)
    pixel_mask | mask to turn spaital maps back to regular grided masked arrays
    cols | number of columns for ifg plot to have.   
    colorbar | 1 or 0 |  1 add colorbar to each ifg, 0 add one for whole figure (if shared is set to 1, too)
    shared | 1 or 0 | 1 and all ifgs share the same colour scale, 0 and don't
    
    DEPENDENCIES:
        
    
    2017/02/17 | modified to use masked arrays that are given as vectors by ifgs, but can be converted back to 
                 masked arrays using the pixel mask    
    2017/05/08 | option to add a colorbar to each ifg
    2017/07/03 | option to make all ifgs share the same scale
    2017/07/03 | function from stack exchange to make 0 of ifgs plot as white. 
    2017/07/06 | fix bug for colourbars that start at 0 and go negative (reverse of above problem)
    2017/08/09 | update so that colourbar is cropped for skewed data (ie -1 to +10 won't have -1 as dark red as +10 is a dark blue)
    2017/10/05 | fix a bug in how the colorbars are plotted
    2017/10/05 | switch to v2
    2017/10/06 | fix a bug in how the colormaps are done when not shared
    2017/10/11 | remove lines redundant after change to remappedColorMap
    
    """ 
    import numpy as np
    import numpy.ma as ma  
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    from small_plot_functions import remappedColorMap
    

    # colour map stuff
    ifg_colours = plt.get_cmap('coolwarm')
    if shared:
        cmap_mid = 1 - np.max(ifgs)/(np.max(ifgs) + abs(np.min(ifgs)))          # get the ratio of the data that 0 lies at (eg if data is -15 to 5, ratio is 0.75)
        print('Cmap centre: ' + str(cmap_mid))
        if cmap_mid > 0.5:
            ifg_colours_cent = remappedColorMap(ifg_colours, start=0.0, midpoint=cmap_mid, stop=(0.5 + (1-cmap_mid)), name='shiftedcmap')
        else:
            ifg_colours_cent = remappedColorMap(ifg_colours, start=(0.5 - cmap_mid), midpoint=cmap_mid, stop=1, name='shiftedcmap')
    
    #data stuff - convert the ifgs from columns to a list of masked arrays
    ifgs_ma = []
    for i in range(np.size(ifgs,1)):
        ifgs_ma.append(ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask ))
        ifgs_ma[i].unshare_mask()
        ifgs_ma[i][~ifgs_ma[i].mask] = ifgs[:,i].ravel()  
    
    # the plot
    pixels, times = ifgs.shape
    rows = int(np.ceil(times/float(cols)))
    f, (ax_all) = plt.subplots(rows, cols, figsize=(14,4))
    f.suptitle(title, fontsize=14)
    ax_all_1d = np.ndarray.flatten(ax_all)
    for temp_ax in ax_all_1d:                   # 
        temp_ax.set_yticks([])
        temp_ax.set_xticks([])
    

    # loop through plotting ifgs
    for i in range(times):
        if shared:
            im = ax_all_1d[i].imshow(ifgs_ma[i], cmap = ifg_colours_cent, vmin=np.min(ifgs), vmax=np.max(ifgs))            # either plot with shared colours
        else: 
            cmap_mid = 1 - np.max(ifgs[:,i])/(np.max(ifgs[:,i]) + abs(np.min(ifgs[:,i])))          # get cmap mid for each ifg
            ifg_cols_local = remappedColorMap(ifg_colours, start=0.0, midpoint=cmap_mid, stop=1, name='ifg_cols_local')         # remappedColorMap function now includes stuff from below
            im = ax_all_1d[i].imshow(ifgs_ma[i], cmap = ifg_cols_local)                                                       # or just own scale
            if colorbar == 1:                                                                                               # and if own scale, might want a colorbar on each plot
                f.colorbar(im, ax=ax_all_1d[i])
    
    #shared colorbar
    if colorbar == 1 and shared == 1:                                                                           # if colormap is shared, only one colorbar needed at edge
        f.subplots_adjust(right=0.8)            
        cbar_ax = f.add_axes([0.95, 0.15, 0.02, 0.7])
        f.colorbar(im, cax=cbar_ax)
        f.tight_layout()
        
    # remove any unused axes
    ax_list = np.ravel(ax_all)              # all axes as list
    ax_list = ax_list[times:]             # ones we want to delete
    for axes in ax_list:                    # loop through deleting them
        f.delaxes(axes)
    f.tight_layout()




#%%

def component_plot(spatial_map, pixel_mask, timecourse, shape, title, shared = 0, temporal_baselines = None):
    """
    Input:
        spatial map | pxc matrix of c component maps (p pixels)
        pixel_mask | mask to turn spaital maps back to regular grided masked arrays
        codings | cxt matrix of c time courses (t long)   
        shape | tuple | the shape of the grid that the spatial maps are reshaped to
        shared | 0 or 1 | if 1, spatial maps share colorbar and time courses shared vertical axis
        Temporal_baselines | x axis values for time courses.  Useful if some data are missing (ie the odd 24 day ifgs in a time series of mainly 12 day)
        
    2017/02/17 | modified to use masked arrays that are given as vectors by spatial map, but can be converted back to 
                 masked arrays using the pixel mask    
    2017/05/12 | shared scales as decrived in 'shared'
    2017/05/15 | remove shared colorbar for spatial maps
    2017/10/16 | remove limit on the number of componets to plot (was 5)
    2017/12/06 | Add a colorbar if the plots are shared, add an option for the time courses to be done in days
    2017/12/?? | add the option to pass temporal baselines to the function
    
    """
    import numpy as np
    import numpy.ma as ma  
    import matplotlib.pyplot as plt
    import matplotlib
    from small_plot_functions import remappedColorMap
    
    def linegraph(sig, ax, temporal_baselines = None):
        """ signal is a 1xt row vector """
        
        if temporal_baselines is None:
            times = sig.size
            a = np.arange(times)
        else:
            a = temporal_baselines
        ax.plot(a,sig,marker='o', color='k')
        ax.axhline(y=0, color='k', alpha=0.4) 
        
 
    
    # colour map stuff
    ifg_colours = plt.get_cmap('coolwarm')
    cmap_mid = 1 - np.max(spatial_map)/(np.max(spatial_map) + abs(np.min(spatial_map)))          # get the ratio of the data that 0 lies at (eg if data is -15 to 5, ratio is 0.75)
    if cmap_mid < (1/257):                                                  # this is a fudge so that if plot starts at 0 doesn't include the negative colorus for the smallest values
        ifg_colours_cent = remappedColorMap(ifg_colours, start=0.5, midpoint=0.5, stop=1.0, name='shiftedcmap')
    else:
        ifg_colours_cent = remappedColorMap(ifg_colours, start=0.0, midpoint=cmap_mid, stop=1.0, name='shiftedcmap')
    
    #make a list of ifgs as masked arrays (and not column vectors)
    spatial_maps_ma = []
    for i in range(np.size(spatial_map,1)):
        spatial_maps_ma.append(ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask ))
        spatial_maps_ma[i].unshare_mask()
        spatial_maps_ma[i][~spatial_maps_ma[i].mask] = spatial_map[:,i].ravel()
    tmp, n_sources = spatial_map.shape
#    if n_sources > 5:
#        n_sources = 5
    del tmp
    
    f, (ax_all) = plt.subplots(2, n_sources, figsize=(15,7))
    f.suptitle(title, fontsize=14)
    f.canvas.manager.set_window_title(title)
    for i in range(n_sources):    
        im = ax_all[0,i].imshow(spatial_maps_ma[i], cmap = ifg_colours_cent, vmin = np.min(spatial_map), vmax = np.max(spatial_map))
        ax_all[0,i].set_xticks([])
        ax_all[0,i].set_yticks([])
#        if shared == 0:
#            ax_all[0,i].imshow(spatial_maps_ma[i])
#        else:
#            im = ax_all[0,i].imshow(spatial_maps_ma[i], vmin = np.min(spatial_map) , vmax =  np.max(spatial_map))
    for i in range(n_sources):
        linegraph(timecourse[i,:], ax_all[1,i], temporal_baselines)
        if temporal_baselines is not None: 
            ax_all[1,i].set_xlabel('Days')
        if shared ==1:
            ax_all[1,i].set_ylim([np.min(timecourse) , np.max(timecourse)])
            
            
    if shared == 1:
        f.tight_layout(rect=[0, 0, 0.94, 1])
        cax = f.add_axes([0.94, 0.6, 0.01, 0.3])
        f.colorbar(im, cax=cax, orientation='vertical')
 



