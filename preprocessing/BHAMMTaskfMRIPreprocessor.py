# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:12:46 2020

@author: rajsri
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
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import concat_imgs, index_img
from nilearn import plotting
import math
import nibabel as nib
import argparse
import preprocextras as pe
import plsextras as plse
import jinja2
from bids.tests import get_test_data_path
import csv


## Set Paths & Make Output Folders
project_path= os.path.join('')
script_path= os.path.dirname(os.path.abspath(__file__))
#Output for PLS batch text file
batchText_output_path=os.path.join('')
if not(os.path.isdir(batchText_output_path)):
    os.mkdir(batchText_output_path)
#QC report path
reports_path=os.path.join('')
if not(os.path.isdir(reports_path)):
    os.mkdir(reports_path)

#Get css template for reports
if not(os.path.islink(os.path.join(reports_path, 'css'))):
    os.symlink(os.path.join(script_path, 'html_report_template', 'css'), os.path.join(reports_path, 'css'), target_is_directory=True)

#Output for Realignment plots
plot_path=os.path.join(reports_path, 'QC_Realignment_Plots')
if not(os.path.isdir(plot_path)):
    os.mkdir(plot_path)

#Output path for final preprocessed nifti files
nifti_output_path=os.path.join('')
if not(os.path.isdir(nifti_output_path)):
    os.mkdir(nifti_output_path)

#Input path for onset files.
onset_path=os.path.join('')
onset_tasks=['ENC_CS_Easy','ENC_Recog_Easy','ENC_Smiss_Easy','ENC_Misses_Easy','ENC_CS_Hard','ENC_Recog_Hard','ENC_Smiss_Hard','ENC_Misses_Hard','RET_CS_Easy','RET_Recog_Easy','RET_Smiss_Easy','RET_Misses_Easy','RET_CR_Easy','RET_FArecog_Easy','RET_FAsmiss_Easy','RET_CS_Hard','RET_Recog_Hard','RET_Smiss_Hard','RET_Misses_Hard','RET_CR_Hard','RET_FArecog_Hard','RET_FAsmiss_Hard','ENC_Fix1_Easy','ENC_Eval_Easy','ENC_Fix2_Easy','RET_Fix1_Easy','RET_Eval_Easy','RET_Fix2_Easy','ENC_Fix1_Hard','ENC_Eval_Hard','ENC_Fix2_Hard','RET_Fix1_Hard','RET_Eval_Hard','RET_Fix2_Hard']


output_path=os.path.join(script_path)

IDs=list(sys.argv[1:])
tasks={'easy1', 'easy2', 'easy3', 'easy4', 'hard1', 'hard2', 'hard3', 'hard4'}
task_timing = {'easy': {'Enc': ['1','20','141','160'], 'Ret': ['58','137','198','277']}, 'hard': {'Enc': ['1','38'], 'Ret':['77','205']}}

combine_cond_dict={'Enc_Hits_Easy': ['ENC_CS_Easy','ENC_Recog_Easy','ENC_Smiss_Easy'], 'Enc_Hits_Hard': ['ENC_CS_Hard','ENC_Recog_Hard','ENC_Smiss_Hard'], 'Ret_Hits_Easy': ['RET_CS_Easy','RET_Recog_Easy','RET_Smiss_Easy'], 'Ret_Hits_Hard': ['RET_CS_Hard','RET_Recog_Hard','RET_Smiss_Hard'] }
combine_cond_dict_load={'Enc_CS': ['ENC_CS_Easy','ENC_CS_Hard'], 'Enc_Recog': ['ENC_Recog_Easy','ENC_Recog_Hard', 'ENC_Smiss_Easy', 'ENC_Smiss_Hard'], 'Ret_CS': ['RET_CS_Easy','RET_CS_Hard'], 'Ret_Recog': ['RET_Recog_Easy','RET_Recog_Hard', 'RET_Smiss_Easy', 'RET_Smiss_Hard'], 'Ret_CR': ['RET_CR_Easy', 'RET_CR_Hard'] }
combine_cond_allOld={'Enc_AllOld_Easy': ['ENC_CS_Easy','ENC_Recog_Easy','ENC_Smiss_Easy', 'ENC_Misses_Easy'], 'Enc_AllOld_Hard': ['ENC_CS_Hard','ENC_Recog_Hard','ENC_Smiss_Hard', 'ENC_Misses_Hard'], 'Ret_AllOld_Easy': ['RET_CS_Easy','RET_Recog_Easy','RET_Smiss_Easy', 'RET_Misses_Easy'], 'Ret_AllOld_Hard': ['RET_CS_Hard','RET_Recog_Hard','RET_Smiss_Hard', 'RET_Misses_Hard'] }
combine_cond_Failure={'Enc_SouceFailure_Easy': ['ENC_Recog_Easy','ENC_Smiss_Easy'], 'Enc_SouceFailure_Hard': ['ENC_Recog_Hard','ENC_Smiss_Hard'], 'Enc_Misses':['ENC_Misses_Easy','ENC_Misses_Hard'], 'Ret_SourceFailure_Easy': ['RET_Recog_Easy','RET_Smiss_Easy'], 'Ret_SourceFailure_Hard': ['RET_Recog_Hard','RET_Smiss_Hard'], 'Ret_Misses':['RET_Misses_Easy','RET_Misses_Hard']}

#Specify confound variables
confound_vars=['csf','white_matter','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']


space_label = 'MNI152NLin2009cAsym'
descriptor = 'preproc'
FD_thresh=1.0
DVars_thresh=2.0
within_Run=False #(set this to true if you want to scrub/smooth/clean within run.)
bad_subj=[]
scrubbed_rows=[]
for ID in IDs:
    print(ID)
    folder = os.path.join(project_path, 'BIDS', 'derivatives',  'fmriprep20.2.0', 'fmriprep','sub-%s' %(str(ID)), "func")
    output_folder=os.path.join(nifti_output_path, 'sub-%s' %(str(ID)))
    if not(os.path.isdir(output_folder)):
        os.mkdir(output_folder)

    file_list= glob.glob("%s/*%s*%s*.nii.gz"%(folder, space_label, descriptor))
    file_list=[x for x in file_list if any(elem in x for elem in tasks)]
    file_list.sort()
    
    mask_file_list= glob.glob("%s/*%s*brain_mask*.nii.gz"%(folder, space_label))
    mask_file_list=[x for x in mask_file_list if any(elem in x for elem in tasks)]
    mask_file_list.sort()

    confound_file_list = glob.glob("%s/*confounds_timeseries.tsv"%(folder))
    confound_file_list=[x for x in confound_file_list if any(elem in x for elem in tasks)]
    confound_file_list.sort()

    #Read Onset file
    try:
        onset_df=pandas.read_excel(os.path.join(onset_path, 'Onsets_fMRI_CIHR2017_subj%s.xlsx' %(str(ID))), sheet_name='SPM PLS ONSETS')
    except:
        print("File not found: %s"%(os.path.join(onset_path, 'Onsets_fMRI_CIHR2017_subj%s.xlsx' %(str(ID)))))
        continue

    totalVolumestillRun=0
    bad_vols_all_runs=[]
    censor_list_all_runs=[]
    trimmed_confounds_all_runs=pandas.DataFrame()
    img_all_runs_list=[]
    report_rows=[]
    report_row_index_names=[]
    meanFD=[]
    # starting & ending index across all runs
    start_ind=0
    end_ind=0
    count_enc_all_runs=0
    count_ret_all_runs=0
    sum_enc_all_runs=0
    sum_ret_all_runs=0
    
    for i,myfile in enumerate(file_list):
        print(myfile)
        try:
            img = nilearn.image.load_img(myfile)
        except:
            print("File not found: %s"%(myfile))
            continue

        try:
            mask_img = nilearn.image.load_img(os.path.join(folder, mask_file_list[i]))
        except:
            print("File not found: %s"%(myfile))
            continue

        try:
            confound_df = pandas.read_csv(os.path.join(folder, confound_file_list[i]), sep="\t")
        except:
            print("File not found: %s"%(confound_file_list[i]))
            continue


        ## Save trimmed_img
        trimmed_img, trimmed_confounds = pe.get_trimmed_img(img,confound_df, scans_to_trim=5)
        
        trimmed_confounds_all_runs=trimmed_confounds_all_runs.append(trimmed_confounds, ignore_index=True)
        
        img_all_runs_list=pe.concat_img_all_runs(trimmed_img, img_all_runs_list)
       
        ## Convert to Time series
        brain_time_series = pe.get_time_series(trimmed_img,mask_img)

        # print(totalVolumestillRun)
        ##Get bad_vols
        bad_vols = pe.get_bad_vols(brain_time_series, trimmed_confounds, FD_thresh=FD_thresh)

        drive, path=os.path.split(myfile)
        path,filename=os.path.split(path)
        filename,file_ext=os.path.splitext(filename)
        filename,file_ext=os.path.splitext(filename)
        task=filename.split('_')[1].split('-')[1]
        run=filename.split('_')[2].split('-')[1]

        if "easy" in task:
            timing=task_timing['easy']
        else:
            timing=task_timing['hard']

        # #Save plot
        fig=pe.get_plot(trimmed_confounds,ID,brain_time_series, timing,FD_thresh=FD_thresh, Dvars_thresh=DVars_thresh, title='sub-%s FD & DVars plot for run-%s, task-%s' %(str(ID), run, task))
        fig.savefig(os.path.join(plot_path, "sub-%s_task-%s_run-%s_FD_DVars_plot.png"%(str(ID),task,run)),facecolor=fig.get_facecolor())



        ##Scrub data
        if len(bad_vols)>0:

            count_enc, count_ret, percent_badVols_enc, percent_badVols_ret, sum_enc, sum_ret= pe.count_bad_vols_task(bad_vols, timing)
            count_enc_all_runs+=count_enc
            count_ret_all_runs+=count_ret
            sum_enc_all_runs+=sum_enc
            sum_ret_all_runs+= sum_ret
            # print("Count Enc Bad Vols: %d. Percent Bad Vols: %f"%(count_enc, percent_badVols_enc))
            # print("Count Ret Bad Vols: %d. Percent Bad Vols: %f"%(count_ret,percent_badVols_ret))
            report_row_index_names.append(task)
            if percent_badVols_enc > 25:
                print("Too many bad volumes! Drop Encoding!")
            if percent_badVols_ret > 25:
                print("Too many bad volumes! Drop Retrieval!")

            report_rows.append([len(bad_vols), count_enc+count_ret, count_enc, round(percent_badVols_enc,2), count_ret, round(percent_badVols_ret,2)])

            bad_vols=[x+totalVolumestillRun for x in bad_vols]
            bad_vols_all_runs.extend(bad_vols)
            # Scrub within run
            if within_Run==True:
                scrubbed_img = pe.scrub_data(trimmed_img, trimmed_confounds,bad_vols)

       

        #Keep track of total volumes so far
        totalVolumestillRun+=brain_time_series.shape[0]

        #Do all this within run (to account for varying columns of DCT regressor)
        # Smooth img
        if within_Run == True:
            cleaned_img = pe.clean_data(scrubbed_img, trimmed_confounds, confound_vars,mask_img)

            print('Smoothing img within run')
            smoothed_img=pe.smooth_data(cleaned_img,fwhm=6)

            #Save 3d scans
            end_ind=end_ind+smoothed_img.shape[3]
            print('Saving 3D volumes By run')
            pe.save_3d_scans_run(smoothed_img, start_ind, end_ind, output_folder,myfile, scans_to_trim=5)
            start_ind=end_ind;

    
    censor_list=[]
    #Concat all runs
    trimmed_img_concat = concat_imgs(img_all_runs_list)
    #Scrub data all runs

    if len(bad_vols_all_runs)>0:
        consec_list=pe.check_consecutive_vols(bad_vols_all_runs)
        censor_list = pe.get_action(bad_vols_all_runs,consec_list)
        if within_Run==False:
            scrubbed_img = pe.scrub_data(trimmed_img_concat, trimmed_confounds_all_runs,bad_vols_all_runs)

    if len(bad_vols_all_runs)==0 and within_Run == False:
        scrubbed_img = trimmed_img_concat

    #Report df
    if len(report_rows)>0:
        report_df = pandas.DataFrame(report_rows, columns=["Total Bad Volumes (all)", "Total Bad Volumes (Enc+Ret)", "No Of Bad Volumes (Enc)", "% Bad Volumes (Enc)", "No Of Bad Volumes (Ret)", "% Bad Volumes (Ret)"])
        report_df.index=report_row_index_names
        # print(report_df.head())
    else:
        report_df=pandas.DataFrame()

    # Smooth/Clean img across all runs
    if within_Run==False:
        print('Smoothing img')
        smoothed_img=pe.smooth_data(scrubbed_img,fwhm=6)
        # #
        # # Remove confound variables
        cleaned_img = pe.clean_data(smoothed_img, trimmed_confounds_all_runs, confound_vars,mask_img)
        #Save 3d scans
        print('Saving 3D volumes')
        pe.save_3d_scans(smoothed_img,output_folder,file_list, scans_to_trim=5)

    # # #Get onsets to censor
    pls_onset_df, remove_dict = plse.censor_onsets(onset_df,rowNum=63, censor_list=censor_list, tasksToInclude=onset_tasks)
    
    onset_to_keep=['RET_CR_Easy', 'RET_CR_Hard']
    new_onset_dict=plse.createNewOnsetDict(pls_onset_df,combine_cond_allOld,onset_to_keep=onset_to_keep,censor_list=censor_list)
    
    # # Make PLS batch text files
    plse.createBatchTxtFiles(new_onset_dict,ID,batchText_output_path, script_path, nifti_output_path, combineSmiss=False)

    ##Generate reports
    ##Get all figures
    fig_list=glob.glob(os.path.join(plot_path,"sub-*%s*.png"%(str(ID))))
    fig_list.sort()
    pe.generate_report(report_df,fig_list, ID, FD_thresh=FD_thresh, DVars_thresh=DVars_thresh, remove_dict=remove_dict, confound_vars=confound_vars,output_path=reports_path, styleCols=["% Bad Volumes (Enc)", "% Bad Volumes (Ret)"])


