import nilearn
import os
from nilearn import image as nimg
import pandas as pd
import numpy as np
from nilearn import plotting
import nibabel as nib
from nilearn import surface
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from natsort import natsorted



# function to generate subject list from BIDS Directory
def subject_list(BIDS_dir, exclude_subs = None):
    """Returns a list of subject IDs from a BIDS Derivatives directory.
       
       Parameters :
       BIDS_dir : str 
           path to BIDS directory
       exclude_subs : {None, list}
           List of subject IDs to exclude from list. 
           If None, returns full list of subject IDs in directory.
           If list, returns list of subject IDs in directory excluding specified IDs.
    """

    BIDS_dir_contents = os.listdir(BIDS_dir)
    subject_list = [sub for sub in BIDS_dir_contents if sub.startswith('sub') and '.' not in sub]
    if (exclude_subs != None):
        subject_list = [sub for sub in subject_list if not any(removed_sub in sub for removed_sub in exclude_subs)]
    subject_list_sorted = natsort(subject_list)
    return(subject_list_sorted)

#function to read in list of nifti objects from list of paths
def load_nifti(img_paths_list, tr_remove = None):
    """Returns a list of nifti objects with first n of brain volumes removed
       
       Parameters :
       img_paths_list : list
           list of paths to nifti files
       tr_remove : {None, int}
           number of brain volumes to remove from beginning of scan.
           If None, returns list of nifti objects without volumes removed.
    """
    assert type(img_paths_list) == list, 'img_list must be of type list'
    img_list =[]
    for ii in range(len(img_paths_list)):
        #read in image
        img = nimg.load_img(img_paths_list[ii])
        if tr_remove != None:
            img_list.append(img.slicer[:,:,:,tr_remove:])
        else:
            img_list.append(img)
    return(img_list)

def load_gifti(img_paths_list, tr_remove = None):
    """Returns a list of nifti objects with first n of brain volumes removed
       
       Parameters :
       img_paths_list : list
           list of paths to nifti files
       tr_remove : {None, int}
           number of brain volumes to remove from beginning of scan.
           If None, returns list of nifti objects without volumes removed.
    """
    assert type(img_paths_list) == list, 'img_list must be of type list'
    img_list =[]
    for ii in range(len(img_paths_list)):
        #read in image
        img = surface.load_surf_data(img_paths_list[ii][0])
        if tr_remove != None:
            img_list.append(np.delete(img,list(range(tr_remove)),axis=1))
        else:
            img_list.append(img)
    return(img_list)

#read in events tsv, remove first n volumes or time, append 
def load_events(events_paths_list, tr_remove = None, tr = None):
    """Returns a list of events dataframes with first n of tr time removed.
       
       Parameters :
       events_paths_list : list
           List of paths to events tsv files.
       tr_remove : {None, int}
           Number brain volumes worth of time to remove from onset, offset, and duration times.
           If None, returns a list of events dataframes without adjusted times.
       tr : {None, float}
           Number of seconds for each tr.
    """
    assert type(events_paths_list) == list, 'events_paths_list must be of type list'
    if tr_remove != None:
        assert tr != None, 'must provide a tr'
    events_data_list = []
    for ii in range(len(events_paths_list)):
        event = pd.read_table(''.join(events_paths_list[ii]))
        if tr_remove != None:
            event.onset = (event.onset - (tr_remove*tr))
        event.duration = event.tTrialEnd - event.tTrialStart
        events_data_list.append(event)
    
    for events in range(len(events_data_list)):
        for entry in range(len(events_data_list[events].trial_type)):
            (events_data_list[events].trial_type[entry]) = (events_data_list[events].trial_type[entry]).lower()
    
    return(events_data_list)

def load_confounds(confounds_paths_list, selected_confounds, tr_remove = None):
    """Returns a list of confound dataframes with first n of tr time removed.
       
       Parameters :
       confounds_paths_list : list
           List of paths to confound tsv files.
       selected_confounds : list
           List of confounds to be included in confound matrix.
       tr_remove : {None, int}
           Number brain volumes worth of time to remove from onset, offset, and duration times.
           If None, returns a list of events dataframes without adjusted times.
    """
    assert type(confounds_paths_list) == list, 'confounds_paths_list must be of type list'
    
    confounds_list = []
    for ii in range(len(confounds_paths_list)):
        confounds = pd.read_table(''.join(confounds_paths_list[ii]))
        if tr_remove != None:
            confounds = confounds.drop(list(range(tr_remove)))
        confounds.drop(confounds.columns.difference(selected_confounds), 1, inplace=True)
        confounds = confounds.reset_index()
        confounds_list.append(confounds)
    return(confounds_list)

def seperate_by_task(img_list, events_list, confounds_list, task_list):
    """Returns lists of nifti imgs, events data frames, and confound data frames seperated by task.
    
        Parameters:
        img_list : list
            List of nifti objects.
        events_list : list
            List of events data frames.
        confounds_list : list
            List of confound estimate data frames.
        task_list : list
            List of task names. Must be of length 2.
    """
    
    assert len(task_list) == 2, 'Current code only works if there are 2 tasks. Adjust function code if you have more than 2 tasks'
    task0_indices =[]
    task1_indices = []

    for ii in range(len(events_list)):
        if events_list[ii].taskName[0] == task_list[0]:
            #print('run-'+str(events_data_list[ii].runNum[0])+'_is_'+events_data_list[ii].taskName[0])
            task0_indices.append(ii)
        elif events_list[ii].taskName[0] == task_list[1]:
            #print('run-'+str(events_data_list[ii].runNum[0])+'_is_'+events_data_list[ii].taskName[0])
            task1_indices.append(ii)
        else:
            print('run-'+str(events_list[ii].runNum[0])+'_is_'+events_list[ii].taskName[0])

    task0_imgs = [img_list[ii] for ii in task0_indices]
    task0_events = [events_list[ii] for ii in task0_indices]
    task0_confounds = [confounds_list[ii] for ii in task0_indices]

    task1_imgs = [img_list[ii] for ii in task1_indices]
    task1_events = [events_list[ii] for ii in task1_indices]
    task1_confounds = [confounds_list[ii] for ii in task1_indices]

    imgs = [task0_imgs,task1_imgs]
    events = [task0_events,task1_events]
    confounds = [task0_confounds,task1_confounds]
    
    return (imgs, events, confounds)

def contrast_cateogries(conditions, category_list, condition_names):
    """Returns an array with each cetegorie's weights for the design matrix.
    
        Paramaters : 
        conditions : dict
            dictionary of all stimulus sub-category weights
        category_list : list
            nested list co categories and sub-categories of stimulus names
        condition_names : list
            list of each stimulus sub-category name
            """
    other_array = np.zeros([len(category_list), len(condition_names)])
    for i in range(len(category_list)):
        for sub_category in category_list[i]:
            other_array[i] += np.array(conditions[sub_category])
        other_array[i] /= len(category_list[i])
    return(other_array)


def contrast(fmri_glm, contrast_list, task, contrast_name_list, sub, map_type = 'stat', stat_type=None, out_dir=None, fig_dir=None, vmax=5, threshold=2.5):
    """
    Returns list of statistical maps of all contrasts computed. Can also save statistical maps as niftis and save figures of contrasts if desired.
    
    Parameters : 
    fmri_glm : model object
        model of data
    contrast_list : list
        list of contrasts to be performed
    task : str
        for naming purposes only
    contrast_name_list : list
        for naming purposes only - list of names on contrasts define by contrast_list
    sub : str
        for naming purposees only - subject id
    map_type : str

    stat_type : str
        't' or 'F'
    out_dir : None or str
        path to save stat map nifti
    fig_dir : None or str
        path to save figure
        
    """
    stat_maps = []
    for ii in range(len(contrast_list)):
        stat_map = fmri_glm.compute_contrast(contrast_list[ii], stat_type = 't', output_type= map_type)
        stat_maps.append(stat_map)
        if out_dir != None:
            map_path = out_dir+map_type+'/'
            if not os.path.exists(map_path):
                os.makedirs(map_path)
            nib.save(stat_map, map_path+sub+'_'+contrast_name_list[ii]+'_task-'+task+'_'+map_type+'.nii.gz')    
    
        if fig_dir !=None:
            fig_path = fig_dir+map_type+'/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plotting.plot_img_on_surf(stat_map,
                          views=['lateral', 'ventral'],
                          hemispheres=['left', 'right'],
                          colorbar=True, 
                          vmax = vmax, threshold=threshold,
                          inflate = True,
                          title = sub+' '+contrast_name_list[ii]+' '+task,
                          output_file = fig_path+sub+'_'+contrast_name_list[ii]+'_task-'+task+'_'+map_type+'.png'
                         )
            plotting.plot_stat_map(stat_map,
                                   vmax = vmax, threshold=threshold,
                                   cut_coords = (-10,-12,-14),
                                   display_mode='z', 
                                   black_bg=True,
                                   title = sub+' '+contrast_name_list[ii]+' '+task,
                                      output_file = fig_path+sub+'_'+contrast_name_list[ii]+'_task-'+task+'_'+map_type+'_volume.png'
                            )
    
    return(stat_maps)

def read_in_rois(label_path_list):
    """Reads in any number of label paths and return a single roi label.
       
       Parameters :
       label_path_list : list 
           list of label paths
    """

    if len (label_path_list)>0:   
        if len(label_path_list) >1: #multiple labels of same roi
            label =surface.load_surf_data(label_path_list[0])
            for roi_indx in range(len(label_path_list)):
                if roi_indx >0:
                    label=np.concatenate((label,surface.load_surf_data(label_path_list[roi_indx])))
        else: #only 1 label per roi
            label =surface.load_surf_data(label_path_list[0])
        return(label)



# Function to fit regression model and annotate with coefficients and p-values
def annotate_regression_OLS(data, x_param, y_param, **kwargs):
    """Calculate ordinary least squares coefficients and add them to plot

        Parameters :
        data : dataframe 
            data frame of data
        x_param : str
            name of df column plotted on x axis
        x_param : str
            name of df column plotted on y axis
        
    """
    
    # Remove rows with NaN values
    data = data.dropna(subset=[x_param, y_param])
    x = data[x_param]
    y = data[y_param]
  
    # Fit regression model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # Check if the model fitting was successful
    if model.params.size > 1:
        coef = model.params[1]
        pval = model.pvalues[1]
    else:
        coef = float('nan')
        pval = float('nan')
    
    # Get axis object
    ax = plt.gca()
    
    # Annotate with coefficients and p-values
    ax.annotate(f'β: {coef:.3f}\np: {pval:.3e}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=14, ha='left', va='top', bbox=dict(facecolor='white', alpha=0))
    # ax.annotate(f'β: {coef:.2f}\np: {pval:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
    #             fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0))

def annotate_regression_corr(data, x_param, y_param,fontsize=20, **kwargs):
    """Calculate Pearson's correlation coefficients and add them to plot

        Parameters :
        data : dataframe 
            data frame of data
        x_param : str
            name of df column plotted on x axis
        x_param : str
            name of df column plotted on y axis
        fontsize : int
            size of annotation font
        
    """
    # Remove rows with NaN values
    data = data.dropna(subset=[x_param, y_param])
    x = data[x_param]
    y = data[y_param]

    # calculate correlation coeficient
    r, p = pearsonr(x, y)



    if p < 0.001:
        significance = '***'
    elif p < 0.01:
        significance = '**'
    elif p < 0.05:
        significance = '*'
    elif p >= 0.05:
        significance = ''


    # Get axis object
    ax = plt.gca()

    # Annotate with coefficients and p-values
    # ax.annotate(f'r: {r:.2f} {significance}', xy=(0.05, 0.95), xycoords='axes fraction', 
    #             fontsize=14, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, fill=False))
    ax.annotate(f'r: {r:.3f}\np: {p:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=fontsize, ha='left', va='top', bbox=dict(facecolor='white', alpha=0))
