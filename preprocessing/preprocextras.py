"""preproc extras module

This module contains helper methods to run post pre-processing steps on task/rs fMRI scans that have already been preprocesed.
@author: Charana (charana.rajagopal@gmail.com) 
"""
import pandas
import numpy
import math
import nilearn
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import glob
import re
from collections import OrderedDict
from datetime import date, datetime
import math
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import concat_imgs, index_img, math_img
import csv
import statistics
import itertools
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import jinja2
import preprocextras as pe
import pdb

def dirname(path):
    return os.path.dirname(path)

def basename(path):
     return os.path.basename(path)

def path_exists(path):
    return os.path.exists(path)

def color_negative_red(value):
    """ Helper for generate_report.
    Colors elements in a dateframe in
    red if >25. Does not color NaN
    values.

    Parameters
    -----------
    value : float
        The value in dataframe to be colored
    
    Returns
    -----------
    color : str
        The color of dataframe value
    """

    if value >= 25.0:
        color = 'red'
    else:
        color = 'black'

    return 'color: %s' % color



def generate_report(report_df, fig_list, ID, FD_thresh=1.0, DVars_thresh=2.0, total_scrubbed_rows=0.0, total_censored_onsets=0, remove_dict=None, confound_vars=None, output_path=os.path.dirname(os.path.abspath(__file__)), styleCols=[]):
    """ Script to generate QC report.
    Creates a html QC report file.

    Parameters
    -----------
    report_df : pandas dataframe
        The dataframe contaning scrubbing related stats
    fig_list : list
        List containing paths of QC realignment plots
    ID : str
        ID of participant
    FD_thresh : float
        FD threshold (optional, default: 1.0)
    DVars_threh : float
        DVars threshold (optional, default: 2.0)
    total_scrubbed_rows : int
        Total number of scrubbed rows across all runs (optional, default: 0)
    total_censored_onsets : int
        Total number of censored onsets (optional, default: 0)
    remove_dict : dictionary
        Dictionary of onsets to be censored (optional, default: None)
    output_path : str
        Path to store output QC reports (optional, default: current path)
    style_cols: list
        Specific column names in report_df for which styling option has to be applied. (optional)
    
    """
    # Template handling
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    env.filters["basename"] = pe.basename
    env.filters["dirname"] = pe.dirname

    d = dict.fromkeys(report_df.select_dtypes('float').columns, "{:.2f}")
    # print(d)
    if not report_df.empty:
        styled_df_html=report_df.style.applymap(color_negative_red, subset=styleCols).format(d) #.render()
    else:
        styled_df_html=pandas.DataFrame()

    template = env.get_template('html_report_template/report_template.html')
    # # print(fig_list)
    data ={
        "ID": str(ID),
        "fig_list": fig_list,
        "fthresh": FD_thresh,
        "dvars_thresh": DVars_thresh,
        "censor_onsets": remove_dict,
        "confounds": confound_vars,
        "total_scrubbed_rows": total_scrubbed_rows,
        "total_censored_onsets": total_censored_onsets
    }

    html = template.render(data, scrubbed_scans_table=report_df, styled_df_str=styled_df_html.to_html(index=False), path_exists=pe.path_exists)

    # html = template.render(data, rows=report_df.to_dict(orient='records'), cols=report_df.columns.to_list(), path_exists=pe.path_exists)

    # Write the HTML file
    # with open('test_taskfMRI_QC_report_%s.html' %(str(ID)), 'w') as f:
        # f.write(html)
    with open(os.path.join(output_path, 'sub-%s_taskfMRI_QC_report.html' %(str(ID))), 'w') as f:
        f.write(html)



def get_plot(trimmed_confounds, ID, brain_time_series, timing=None, FD_thresh=1.0, Dvars_thresh=2.0, title=None):
    """ Script to create FD/DVars plots for QC report.

    Parameters
    -----------
    trimmmed_confounds : pandas dataframe
        The dataframe contaning confounds from fmriprep's confounds.tsv
    ID : str
        ID of participant
    brain_time_series: nilearn time-series
        nifti scan converetd to nilearn time-series
    timing : dictionary
        Timing of enc/ret blocks (optional)
    FD_thresh : float
        FD threshold (optional, default: 1.0)
    DVars_threh : float
        DVars threshold (optional, default: 2.0)
    title : str
        Title of plot (optional)
    
     Returns
    -----------
    fig : matplotlib figure
        The matplotlib figure containing plot
    
    """
    ## NOTE: Remove ID & brain_time_series. They are not being used.
    plt.close('all')
    if timing is not None:
        enc_timing=list(timing['Enc'])
        ret_timing=list(timing['Ret'])
    

    matplotlib.rc('axes',edgecolor='w')
    fig = plt.figure()
    # fig.patch.set_facecolor('#e5e5e5')
    fig.patch.set_facecolor((0.5,0.5,0.5))
    fig.suptitle(title, y=.98, size='x-large', fontweight='bold', color='white')
    t = trimmed_confounds.index


    #Plot Dvars
    ax1 = fig.add_subplot(211)
    ax1.axhline(Dvars_thresh, lw=.5, label='%s'%(str(Dvars_thresh)),color='r', linestyle='-', alpha=.75)
    ax1.set_facecolor((0.5,0.5,0.5))
    ax1.grid()
    ax1.set(ylabel='DVars')
    ax1.yaxis.label.set_color('white')
    ax1.get_xaxis().set_visible(False)
    ax1.tick_params(color="white")
    ax1.tick_params(axis='y', colors='white')
    for t in ax1.xaxis.get_ticklines(): t.set_color('white')
    for t in ax1.yaxis.get_ticklines(): t.set_color('white')
    trimmed_confounds.plot(y='std_dvars', ax=ax1, legend=False)

    #Plot FD
    ax2 = fig.add_subplot(212)
    ax2.axhline(FD_thresh, lw=.5, label='%s'%(str(FD_thresh)),color='r', linestyle='-', alpha=.75)
    ax2.set_facecolor((0.5,0.5,0.5))
    ax2.grid()
    ax2.set(ylabel='FD (mm)')
    ax2.yaxis.label.set_color('white')
    ax2.xaxis.label.set_color('white')
    ax2.tick_params(color="white")
    ax2.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='x', colors='white')
    for t in ax2.xaxis.get_ticklines(): t.set_color('white')
    for t in ax2.yaxis.get_ticklines(): t.set_color('white')
    if timing is not None:
        for i in range(0,len(enc_timing),2):
            ax1.axvspan(int(enc_timing[i]), int(enc_timing[i+1]), facecolor='#00BD72', alpha=0.5,zorder=-100)
            ax2.axvspan(int(enc_timing[i]), int(enc_timing[i+1]), facecolor='#00BD72', alpha=0.5,zorder=-100)

        for j in range(0, len(ret_timing),2):
            ax1.axvspan(int(ret_timing[j]), int(ret_timing[j+1]), facecolor='#ffa500', alpha=0.5,zorder=-100)
            ax2.axvspan(int(ret_timing[j]), int(ret_timing[j+1]), facecolor='#ffa500', alpha=0.5,zorder=-100)

        #Create custom legend
        legend_elements = [Patch(facecolor='#00BD72', label='Enc'),
                        Patch(facecolor='#ffa500', label='Ret')]

        fontP = FontProperties()
        fontP.set_size('x-small')


        lg = ax1.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
        lg.get_frame().set_alpha(None)
        lg.get_frame().set_facecolor((0.5,0.5,0.5, 0.1))
        for text in lg.get_texts():
            text.set_color("white")

    #Plot FD
    trimmed_confounds.plot(y='framewise_displacement', ax=ax2, legend=False)#, color='#420420')


    return fig

def get_bad_vols(brain_time_series, trimmed_confounds, FD_thresh=1.0, DVars_thresh=2.0):
    """ Get list of high motion scans (>FD threshold).

    Parameters
    -----------
    brain_time_series: nilearn time-series
        nifti scan converetd to nilearn time-series
    trimmmed_confounds : pandas dataframe
        The dataframe contaning confounds from fmriprep's confounds.tsv
    ID : str
        ID of participant
    FD_thresh : float
        FD threshold (optional, default: 1.0)
    DVars_threh : float
        DVars threshold (optional, default: 2.0)
    
     Returns
    -----------
    a : list
        List containing scan numbers of high motion scans
    
    """
    a=[]
    for i in range(0,len(brain_time_series)):
        # TS= brain_time_series[i]


        ## NOTE: Use motion_outlier_XX columns of confounds.tsv to narrow down high motion scans and then filter based on our threshold
        outlier_cols = [c for c in trimmed_confounds.columns.values if "motion_outlier" in c]

        volumes=[]

        
        for col in outlier_cols:
            index = trimmed_confounds.index[trimmed_confounds[col]==1]
            # print(index)
            if len(index)>0:
                volumes= volumes+index.tolist()

        if i==0:
            no_of_Bad_Vols=0
        for v in volumes:
            if round(trimmed_confounds['framewise_displacement'].iloc[v],1)>FD_thresh:
                if i==0:
                    a.append(v)
                    no_of_Bad_Vols+=1


    return a

def saveCensorOnsets(censor_list, ID, outputFileName, output_path=os.getcwd(),overWrite=False):
    """ Save List of onsets to be censored from PLS batch text files to a csv file.

    Parameters
    -----------
    censor_list: list
        List of onsets to be censored
    ID : str
        ID of participant
    outputFileName : str
        Name of output file
    output_path : str
        Path to save output file (optional, default: current path)
    overWrite : bool
        If True overwrite existing file.  (optional, default: False)
    
    """
    ## NOTE: move to plsextras maybe ?
    fullOutputFile=os.path.join(output_path, outputFileName)
    if not(os.path.isfile(fullOutputFile)) or overWrite==True:
        with open(fullOutputFile,'w+') as f1:
            writer=csv.writer(f1, delimiter=',')#lineterminator='\n',
            writer.writerow(['ID', 'censor_list'])

    with open(fullOutputFile,'a+') as f1:
        censor_list.insert(0,ID)
        writer=csv.writer(f1, delimiter=',')#lineterminator='\n',
        writer.writerow(censor_list)

def get_action(bad_vols, consec_list):
    """ Get what actions needs to be done for high volume scans (scrubbing or censoring)
    
    Helper method for censoring scans
    Mostly to record what needs to be done for logs.
    Also, returns a list of 3 or more consecutive (in time) high motion scans for potential censoring

    Parameters
    -----------
    bad_vols: list
        List of high motion scans  (>FD threshold)
    consec_list: list
        List of high motion scans that consecutive in time (subset of bad_vols)
    
    Returns
    -----------
    consec3_list: list
        List of 3 or more consecutive scans to cheeck for potential censoring

    
    """
    consec3_list=[]
    index=0
    for i in range(len(consec_list)):
        if consec_list[i] == 1:
            print("Scrub vol:", bad_vols[index])
            index+=1
        if consec_list[i] == 2:
            print("Scrub 2 vol", bad_vols[index:index+consec_list[i]])
            index+=2
        if consec_list[i] > 2:
            print("Check if vols have to be censored", bad_vols[index:index+consec_list[i]])
            consec3_list.extend(bad_vols[index:index+consec_list[i]])
            index+=consec_list[i]

    return consec3_list



def check_consecutive_vols(bad_vols):
    """ Check if high motion scans are consecutive in time
    
    Helper method for censoring/scrubbing.
    Returns a subset of 2 or more consecutive (in time) high motion scans.

    Parameters
    -----------
    bad_vols: list
        List of high motion scans  (>FD threshold)
    
    Returns
    -----------
    retlist: list
        List of 2 or more consecutive scans to check for potential scrubbing or censoring

    """
    count=1
    retlist=[]
    # retlist = [1] * len(bad_vols)
    # Avoid IndexError for  random_list[i+1]
    for i in range(len(bad_vols) - 1):
        # Check if the next number is consecutive
        if bad_vols[i] + 1 == bad_vols[i+1]:
            count += 1
        else:
            # If it is not append the count and restart counting
            retlist.append(count)
            # retlist[i] = count
            count = 1
    # Since we stopped the loop one early append the last count
    retlist.append(count)
    # retlist[i+1] = count
    return retlist

def count_bad_vols_task(badvols,task_timing):
    """ Count high motion scans. For QC report tables
    
    Parameters
    -----------
    bad_vols: list
        List of high motion scans  (>FD threshold)
    task_timing: dictionary
        Dictionary containing timing of enc/ret blocks
    
    Returns
    -------------
    count_enc: int
        Number of high motion scans in Enc block
    count_ret: int
        Number of high motion scans in Ret block
    percent_badVols_enc: float
        % of high motion scans in Enc block
    percent_badVols_ret: float
        % of high motion scans in Ret block
    sum_enc: int
        Total number of high motion scans in Enc block
    sum_ret: int
        Total number of high motion scans in Ret block
    
    """
    enc_timing=list(task_timing['Enc'])
    ret_timing=list(task_timing['Ret'])

    count_enc=0
    count_ret=0
    #count_other=0
    sum_enc=0
    sum_ret=0
    percent_badVols_enc=0.0
    percent_badVols_ret=0.0
    for i in range(0,len(enc_timing),2):
        sum_enc+=len(range(int(enc_timing[i]), int(enc_timing[i+1])+1))
        sum_ret+=len(range(int(ret_timing[i]), int(ret_timing[i+1])+1))
        for v in badvols:
            if v in range(int(enc_timing[i]), int(enc_timing[i+1])+1):
                count_enc=count_enc+1

            if v in range(int(ret_timing[i]), int(ret_timing[i+1])+1):
                count_ret=count_ret+1


    # print("Sum enc vols: ",sum_enc)
    # print("Sum ret vols: ", sum_ret)
    if count_enc >0: percent_badVols_enc=round((count_enc/sum_enc)*100.0, 2)
    if count_ret >0: percent_badVols_ret=round((count_ret/sum_ret)*100.0, 2)
    return count_enc, count_ret ,percent_badVols_enc, percent_badVols_ret, sum_enc, sum_ret

def drop_volumes(trimmed_img,vols_to_drop):
    """ Drop scans
    
    Parameters
    -----------
    trimmed_img: nilearn niimg-like object
        Input nifti image in nilearn niimg object
    vols_to_drop: list
        List of scans (in scan number) to be dropped
    
    Returns
    -------------
    img_without_badVols: nilearn niimg-like object
        Output nifti niimg object with scans dropped
    """
    sub_img=list()
    for i in range(0,trimmed_img.shape[3]):
        if i not in vols_to_drop:
            sub_img.append(index_img(trimmed_img, i))
        else:
            continue

    img_without_badVols=concat_imgs(sub_img)

    return img_without_badVols

def get_time_series(img,mask_img=None, t_r=2.0):
    """ Convert Nifti Image   to time series
    
    Helper method for calculating scan numbers of high motion scans
    
    Parameters
    -----------
    img: Nilearn niimg-like object
        Input nifti image
    mask_img: Nilearn niimg-like object
        Mask img from fmriprep outputs
    t_r: float
        TR of fMRI acquistions (optional, default: 2.0)
    
    Returns
    -----------
    brain_time_series: nilearn time-series
        Brain time-series
    """
    # Convert to Time series
    brain_masker = NiftiMasker(t_r=t_r, standardize=False, mask_img=mask_img)
    brain_time_series = brain_masker.fit_transform(img)

    return brain_time_series

def concat_img_all_runs(trimmed_img, img_all_runs_list):
    """ Concatenate scans across multiple runs

    Parameters
    -----------
    trimmed_img: Nilearn niimg-like object
        Input nifti image
    img_all_runs_list: List
        List with concatenated scans
    
    Returns
    -----------
    img_all_runs_list: List
        List with concatenated scans after appending 
    """

    for i in range(0, trimmed_img.shape[3]):
        img_all_runs_list.append(index_img(trimmed_img, i))

    return img_all_runs_list
    # trimmed_img= concat_imgs(sub_imgs)

def get_trimmed_img(img, confound_df, scans_to_trim=5):
    """ Trim first n scans 

    Parameters
    -----------
    img: Nilearn niimg-like object
        Input nifti image
    confound_df: pandas dataframe
        Dataframe contaning the confounds.tsv from fmriprep outputs
    scans_to_trim: int
        Number of scans to be trimmed from the beginning of image
    
    Returns
    -----------
    trimmed_img: Nilearn niimg-like object
        Trimmed Nifti img 
    """
    # regressors= [c for c in confound_df.columns.values if not("aroma" in c)]
    regressors= [c for c in confound_df.columns.values]
    confounds=confound_df[regressors].copy()

    # Trim Nifti - Remove first 5 scans
    sub_imgs=list()

    for i in range(scans_to_trim, img.shape[3]):
        if (i<=confounds.shape[0]):
            sub_imgs.append(index_img(img, i))
        else:
            break

    trimmed_img= concat_imgs(sub_imgs)
    trimmed_confounds= confounds.iloc[scans_to_trim:]
    trimmed_confounds.reset_index(drop=True, inplace=True)


    return trimmed_img, trimmed_confounds

def smooth_data(trimmed_img, fwhm=6):
    """ Smooth nifti image with a Gaussian kernel

    Parameters
    -----------
    trimmed_img: Nilearn niimg-like object
        Input nifti image
    fwhmm: float
        FWHM of Gaussian kernel (optional, default: 6)
    
    Returns
    -----------
    smoothed_img: Nilearn niimg-like object
        Smoothed Nifti img 
    """
    smoothed_img = nilearn.image.smooth_img(trimmed_img, fwhm=fwhm)
    return smoothed_img

def save_3d_scans_run(img, start_ind, end_ind, output_path, myfile, scans_to_trim=5):
    """ Save Nifti scans as 3D nii files by run

    Parameters
    -----------
    img: Nilearn niimg-like object
        Input nifti image
    start_ind: float
        Index of first scan in run
    end_ind: float
        Index of last scan in run
    output_path: str
        Path to save 3D nii scans
    myfile: str
        Filename of input nifti image
    scans_to_trim: int
        Number of scans to trim from the beginning of run
   
    """
    #create_filename
    no_of_total_vols=img.shape[3]
    # print(no_of_total_vols)
    print("Start=%d" %(start_ind))
    print("End=%d" %(end_ind))
    
    try:
        temp_img = nilearn.image.load_img(myfile)
    except:
        print("File not found: %s"%(myfile))

    drive, path=os.path.split(myfile)
    path,filename=os.path.split(path)
    filename,file_ext=os.path.splitext(filename)
    filename,file_ext=os.path.splitext(filename)
    
    print("end_ind-start_ind=%d" %(end_ind-start_ind))
    for i in range(0, no_of_total_vols):

        img_3d=index_img(img,i) # to get scan number within this run & not across all runs
        output_filename='_'.join(filename.split('_')[:-2])+'_desc-smooth_bold_%04d.nii'%(int(i+start_ind))
        # print(output_filename)
        img_3d.to_filename(os.path.join(output_path, output_filename))


def save_3d_scans(img, output_path, file_list=[], count_end_bad_vols = 0, scans_to_trim=5):
    """ Save Nifti scans as 3D nii files across all runs

    Parameters
    -----------
    img: Nilearn niimg-like object
        Input nifti image
    output_path: str
        Path to save 3D nii scans
    file_list: str
        List of filenames of all runs
    count_end_bad_vols: int
        Number of scans to be dropped from end of run
    scans_to_trim: int
        Number of scans to trim from the beginning of run
   
    """
    #create_filename
    no_of_total_vols=img.shape[3]
    print(no_of_total_vols)
    start_ind=0
    end_ind=0
    # no_of_vols=0
    for i, myfile in enumerate(file_list):

        # print(myfile)
        try:
            temp_img = nilearn.image.load_img(myfile)
        except:
            print("File not found: %s"%(myfile))

        drive, path=os.path.split(myfile)
        path,filename=os.path.split(path)
        filename,file_ext=os.path.splitext(filename)
        filename,file_ext=os.path.splitext(filename)
        # print(filename)
        if count_end_bad_vols > 0 and i == len(file_list)-1:
            no_of_vols = temp_img.shape[3] - scans_to_trim - count_end_bad_vols
        else:
            no_of_vols = temp_img.shape[3] - scans_to_trim


        end_ind=end_ind+no_of_vols
        # print(start_ind)
        # print(end_ind)
        for i in range(start_ind,end_ind):
            if end_ind>no_of_total_vols:
                break
            # else:
            img_3d=index_img(img,i)
            output_filename='_'.join(filename.split('_')[:-2])+'_desc-smooth_bold_%04d.nii'%(int(i))
            # print(output_filename)
            img_3d.to_filename(os.path.join(output_path, output_filename))
            start_ind=end_ind


def scrub_boundary_vols(trimmed_img,trimmed_confounds,bad_vols=None):
    """ Drop high motion scans from the end of run

    Parameters
    -----------
    trimmed_img: Nilearn niimg-like object
        Input nifti image
    trimmed_confounds: pandas dataframe 
        Dataframe containing the confounds from confounds.tsv file of fmriprep output
    bad_vols: list
        List containing hight motion scans (>FD threshold)
    
    Returns
    -----------
    trimmed_dropped_img: Nilearn niimg-like object
        Nifti img with high motions scans at the end dropped
    trimmed_confounds: pandas dataframe 
        trimmed confound dataframe
    count_end_bad_vols: int
        Number of high motion scans from the end to be dropped.
    """
    count_end_bad_vols=0
    # isInBadVols=True
    end_idx=trimmed_img.shape[3]-1
    while(True):
        if end_idx in bad_vols:
            count_end_bad_vols +=1
            end_idx -= 1
        else:
            break
    # print(count_end_bad_vols)
    if count_end_bad_vols == 0:
        trimmed_dropped_img = trimmed_img
        
    else:
        if count_end_bad_vols > 5:
            sys.exit("Error! Too many volumes at the end to drop. Delete run!")
        else:
            trimmed_dropped_img = index_img(trimmed_img,slice(0, trimmed_img.shape[3]-count_end_bad_vols))
            trimmed_confounds = trimmed_confounds.head(-count_end_bad_vols)
    
    return trimmed_dropped_img, trimmed_confounds, count_end_bad_vols


def scrub_data(trimmed_img,trimmed_confounds,bad_vols=None):
    """ Scrub high motion scans (>FD threshold)

    Scans that are high motion (>FD threshold) are replaced with the average of the scan before and the scan after.
    Takes into account 2 consectivee high motion scans.
    If there are 3 or more consecutive high motion scans, please look at censoring.

    Parameters
    -----------
    trimmed_img: Nilearn niimg-like object
        Input nifti image
    trimmed_confounds: pandas dataframe 
        Dataframe containing the confounds from confounds.tsv file of fmriprep output
    bad_vols: list
        List containing hight motion scans (>FD threshold)
    
    Returns
    -----------
    img_without_badVols: Nilearn niimg-like object
        Nifti img with high motions scans scrubbed
    trimmed_confounds: pandas dataframe 
        trimmed confound dataframe
    count_end_bad_vols: int
        Number of high motion scans from the end to be dropped.
    """
    ## Check if there are high motions scans at end of last run to be dropped
    trimmed_dropped_img, trimmed_confounds, count_end_bad_vols = scrub_boundary_vols(trimmed_img,trimmed_confounds,bad_vols)
    
    sub_img=list()
    for i in range(0,trimmed_dropped_img.shape[3]):
        if i not in bad_vols:
            sub_img.append(index_img(trimmed_dropped_img, i))
        else:
            if (i-1) < 0:
                img_before=index_img(trimmed_dropped_img,i)
            else:
                img_before=index_img(trimmed_dropped_img,i-1)
            if (i+1) >= trimmed_dropped_img.shape[3]:
                img_after=index_img(trimmed_dropped_img,i)
            else:
                img_after=index_img(trimmed_dropped_img,i+1)

            if (i-1) in bad_vols and (i-1) > 0:
                img_before = index_img(trimmed_dropped_img,i-2)
            if (i+1) in bad_vols and (i+1) < trimmed_dropped_img.shape[3]:
                img_after = index_img(trimmed_dropped_img,i+2)
            fixed_img=math_img("(img1 + img2)/2.0", img1=img_before, img2=img_after)
            sub_img.append(fixed_img)
            # continue

    img_without_badVols=concat_imgs(sub_img)

    return img_without_badVols,trimmed_confounds, count_end_bad_vols

def clean_data(trimmed_img, trimmed_confounds, confound_vars=None, mask_img=None,high_pass=None,low_pass=None,t_r=2.0, type="task"):
    """ Regress confounds from image

    Regress out selected confounds from the confounds.tsv from signal/img

    Parameters
    -----------
    trimmed_img: Nilearn niimg-like object
        Input nifti image
    trimmed_confounds: pandas dataframe 
        Dataframe containing the confounds from confounds.tsv file of fmriprep output
    confound_vars: list
        List containing column names from confounds.tsv of confound variables to be regressed out
    mask_img:  Nilearn niimg-like object
        Input mask nifti image
    high_pass: float
        High cut off frequencyy in Hertz (optional)
    low_pass: float
        Low cut off frequencyy in Hertz (optional)
    t_r: float
        TR of fMRI acquistion in seconds (optional, default: 2.0)
    type: {'task', 'rest'}
        Type of fMRI scan to clean
    
    Returns
    -----------
    cleaned_img: Nilearn niimg-like object
        Nifti img with confounds regressed out
    
    """
    ## Get  only confounds to regress
    trimmed_confounds = trimmed_confounds[confound_vars]
    if type.lower()=="task":
        brain_masker = NiftiMasker(t_r=t_r, standardize=False, mask_img=mask_img, mask_strategy='epi')
    elif type.lower()=="rest":
        brain_masker = NiftiMasker(t_r=t_r, standardize=True, mask_img=mask_img, high_pass=high_pass, low_pass=low_pass,  mask_strategy='epi')
    else:
        sys.exit("Error while running clean_data. Type can only be task or rest")

    brain_masker.fit()

    cleaned_signal=brain_masker.transform(trimmed_img,confounds=trimmed_confounds.values)
    cleaned_img=brain_masker.inverse_transform(cleaned_signal)

    # cleaned_img=nilearn.image.clean_img(trimmed_img, confounds=trimmed_confounds.values, standardize=False, t_r=t_r, mask_img=mask_img)


    return cleaned_img
