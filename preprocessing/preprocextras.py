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
  """
  Colors elements in a dateframe
  green if positive and red if
  negative. Does not color NaN
  values.
  """

  if value >= 25.0:
    color = 'red'
  else:
    color = 'black'

  return 'color: %s' % color



def generate_report(report_df, fig_list, ID, FD_thresh=1.0, DVars_thresh=2.0, remove_dict=None, confound_vars=None, output_path=os.path.dirname(os.path.abspath(__file__)), styleCols=[]):
    # Template handling
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    env.filters["basename"] = pe.basename
    env.filters["dirname"] = pe.dirname

    d = dict.fromkeys(report_df.select_dtypes('float').columns, "{:.2f}")
    # print(d)
    if not report_df.empty:
        styled_df_html=report_df.style.applymap(color_negative_red, subset=styleCols).format(d).render()
    else:
        styled_df_html=None

    template = env.get_template('html_report_template/report_template.html')
    # # print(fig_list)
    data ={
        "ID": str(ID),
        "fig_list": fig_list,
        "fthresh": FD_thresh,
        "dvars_thresh": DVars_thresh,
        "censor_onsets": remove_dict,
        "confounds": confound_vars
    }

    html = template.render(data, scrubbed_scans_table=report_df, styled_df_str=styled_df_html, path_exists=pe.path_exists)

    # html = template.render(data, rows=report_df.to_dict(orient='records'), cols=report_df.columns.to_list(), path_exists=pe.path_exists)

    # Write the HTML file
    # with open('taskfMRI_QC_report_%s.html' %(str(ID)), 'w') as f:
    #     f.write(html)
    with open(os.path.join(output_path, 'sub-%s_taskfMRI_QC_report.html' %(str(ID))), 'w') as f:
        f.write(html)



def get_plot(trimmed_confounds, ID, brain_time_series, timing=None, FD_thresh=1.0, Dvars_thresh=2.0, title=None):
        plt.close('all')
        if timing is not None:
            enc_timing=list(timing['Enc'])
            ret_timing=list(timing['Ret'])
        #
        # if filename == '':
        #     task=""
        #     run=""
        # else:
        #     task=filename.split('_')[1].split('-')[1]
        #     run=filename.split('_')[2].split('-')[1]

        matplotlib.rc('axes',edgecolor='w')
        fig = plt.figure()
        # fig.patch.set_facecolor('#e5e5e5')
        fig.patch.set_facecolor((0.5,0.5,0.5))
        fig.suptitle(title, y=.98, size='x-large', fontweight='bold', color='white')
        t = trimmed_confounds.index



        #Plot Dvars
        ax1 = fig.add_subplot(211)
        ax1.axhline(Dvars_thresh, lw=.5, label='%s'%(str(Dvars_thresh)),color='r', linestyle='-', alpha=.75)
#        fig.text(numpy.median(t), (Dvars_thresh), 'threshold= %s'%(str(Dvars_thresh)), ha='center', va='center', color='w')
        # ax1.set_facecolor('#e5e5e5')
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
        # ax2.set_facecolor('#e5e5e5')
#        fig.text(numpy.median(t), (FD_thresh), 'threshold= %s'%(str(FD_thresh)), ha='center', va='center', color='w')
        ax2.set_facecolor((0.5,0.5,0.5))
        ax2.grid()
	#title='%s Framewise Displacement'%(str(ID))
        ax2.set(ylabel='FD (mm)')
        ax2.yaxis.label.set_color('white')
        ax2.xaxis.label.set_color('white')
#        plt.ylabel('Framewise-Displacement (mm)', size='medium',color="white")
#        ax2.get_xaxis().set_visible(False)
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
    a=[]
    for i in range(0,len(brain_time_series)):
        TS= brain_time_series[i]


        outlier_cols = [c for c in trimmed_confounds.columns.values if "motion_outlier" in c]

        volumes=[]

        #print trimmed_confounds.index.values
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
    censor_list=[]
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
            censor_list.extend(bad_vols[index:index+consec_list[i]])
            index+=consec_list[i]

    return censor_list



def check_consecutive_vols(bad_vols):
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
    sub_img=list()
    for i in range(0,trimmed_img.shape[3]):
        if i not in vols_to_drop:
            sub_img.append(index_img(trimmed_img, i))
        else:
            continue

    img_without_badVols=concat_imgs(sub_img)

    return img_without_badVols

def get_time_series(img,mask_img=None, t_r=2.0):
    # Convert to Time series
    brain_masker = NiftiMasker(t_r=t_r, standardize=True, mask_img=mask_img)
    brain_time_series = brain_masker.fit_transform(img)

    return brain_time_series

def concat_img_all_runs(trimmed_img, img_all_runs_list):

    for i in range(0, trimmed_img.shape[3]):
        img_all_runs_list.append(index_img(trimmed_img, i))

    return img_all_runs_list
    # trimmed_img= concat_imgs(sub_imgs)

def get_trimmed_img(img, confound_df, scans_to_trim=5):
    regressors= [c for c in confound_df.columns.values if not("aroma" in c)]
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
    smoothed_img = nilearn.image.smooth_img(trimmed_img, fwhm=fwhm)
    return smoothed_img

def save_3d_scans_run(img, start_ind, end_ind, output_path, myfile, scans_to_trim=5):
    #create_filename
    no_of_total_vols=img.shape[3]
    # print(no_of_total_vols)
    print("Start=%d" %(start_ind))
    print("End=%d" %(end_ind))
    # start_ind=0
    # end_ind=0
    # no_of_vols=0
    try:
        temp_img = nilearn.image.load_img(myfile)
    except:
        print("File not found: %s"%(myfile))

    drive, path=os.path.split(myfile)
    path,filename=os.path.split(path)
    filename,file_ext=os.path.splitext(filename)
    filename,file_ext=os.path.splitext(filename)
    # print(filename)
    # no_of_vols = temp_img.shape[3] - scans_to_trim

    # if 'easy' in filename:
    #     no_of_vols=282
    # if 'hard' in filename:
    #     no_of_vols=208


    # end_ind=end_ind+no_of_vols;
    # print(start_ind)
    # print(end_ind)
    print("end_ind-start_ind=%d" %(end_ind-start_ind))
    for i in range(0, no_of_total_vols):
        # if (end_ind-start_ind)>no_of_total_vols:
        #     break
        # else:

        img_3d=index_img(img,i) # to get scan number within this run & not across all runs
        output_filename='_'.join(filename.split('_')[:-2])+'_desc-smooth_bold_%04d.nii'%(int(i+start_ind))
        # print(output_filename)
        img_3d.to_filename(os.path.join(output_path, output_filename))


def save_3d_scans(img, output_path, file_list=[], scans_to_trim=5):
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
        no_of_vols = temp_img.shape[3] - scans_to_trim

        # if 'easy' in filename:
        #     no_of_vols=282
        # if 'hard' in filename:
        #     no_of_vols=208


        end_ind=end_ind+no_of_vols;
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
            start_ind=end_ind;

def IndOutOfBound(img, index):
    if index < 0:
        return index+1
    if index >= img.shape[3]:
        index -1

def scrub_data(trimmed_img,trimmed_confounds,bad_vols=None):
    # print(bad_vols)
    sub_img=list()
    for i in range(0,trimmed_img.shape[3]):
        if i not in bad_vols:
            sub_img.append(index_img(trimmed_img, i))
        else:
            if (i-1) < 0:
                img_before=index_img(trimmed_img,i)
            else:
                img_before=index_img(trimmed_img,i-1)
            if (i+1) >= trimmed_img.shape[3]:
                img_after=index_img(trimmed_img,i)
            else:
                img_after=index_img(trimmed_img,i+1)

            if (i-1) in bad_vols and (i-1) > 0:
                img_before = index_img(trimmed_img,i-2)
            if (i+1) in bad_vols and (i+1) < trimmed_img.shape[3]:
                img_after = index_img(trimmed_img,i+2)
            fixed_img=math_img("(img1 + img2)/2.0", img1=img_before, img2=img_after)
            sub_img.append(fixed_img)
            # continue

    img_without_badVols=concat_imgs(sub_img)

    return img_without_badVols

def clean_data(trimmed_img, trimmed_confounds, confound_vars=None, mask_img=None,high_pass=None,t_r=2.0,fwhm=6):

    trimmed_confounds = trimmed_confounds[confound_vars]
    
    # trimmed_confounds
    # Convert to Time series
    brain_masker = NiftiMasker(t_r=t_r, standardize=False, mask_img=mask_img, mask_strategy='epi', smoothing_fwhm=fwhm)
    # brain_masker = NiftiMasker(t_r=t_r, standardize=False, mask_img=mask_img, high_pass=high_pass, low_pass=0.08, detrend=False, smoothing_fwhm=fwhm)

    brain_masker.fit()

    cleaned_signal=brain_masker.transform(trimmed_img,confounds=trimmed_confounds.values)
    cleaned_img=brain_masker.inverse_transform(cleaned_signal)

    # cleaned_img=nilearn.image.clean_img(trimmed_img, confounds=trimmed_confounds.values, standardize=False, t_r=t_r, mask_img=mask_img)


    return cleaned_img
