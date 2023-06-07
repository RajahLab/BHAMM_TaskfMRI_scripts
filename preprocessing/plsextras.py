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
from collections import OrderedDict, defaultdict
from datetime import date, datetime
import math
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import concat_imgs, index_img, math_img
import csv
import statistics
import itertools
import preprocextras as pe

def createNewOnsetDict(pls_onset_df, combine_onset_dict, onset_to_keep=[], censor_list=[]):

    new_onset_dict=dict()
    for newCond,oldCond in combine_onset_dict.items():
        only_values=[value for key,value in pls_onset_df.items() if key in oldCond]
        # all_keys=[key for key,value in pls_onset_df.items() if key in oldCond]
        # print(all_keys)
        only_values=list(itertools.chain.from_iterable(only_values))
        # print(only_values)
        new_onset_dict[newCond] = only_values

    if len(onset_to_keep)>0:
        for cond in onset_to_keep:
            new_onset_dict[cond]=pls_onset_df[cond]

    return new_onset_dict

def createBatchTxtFiles(pls_onset_df,ID,output_path=os.getcwd(), script_path=os.getcwd(), nifti_path=os.getcwd(), combineSmiss=False, default_nifti_path=True, img_format='nii'):
    onset_tasks=[key for key,value in pls_onset_df.items()]
    # print(onset_tasks)
    ## Write into batch_fmri_subj*.txt files
    output_file=os.path.join(output_path,'batch_fmri_subj'+ID+'.txt');
    with open(os.path.join(script_path,'batch_fmri_subjXXX.txt')) as f:
        content=f.readlines()

    if img_format.lower() != 'nii' and img_format.lower() != 'img':
        sys.exit('img_format can only be nii or img')



    content[7]='prefix          subj'+ID+' % prefix for session file and datamat file\n'
    ## put this in a variable.
    if default_nifti_path == False:
        content[41]='data_files     '+nifti_path+'/*.'+img_format+' % run 1 data pattern (must use wildcard)\n\n';
    else:
        content[41]='data_files     '+nifti_path+'/sub-'+ID+'/sub*.'+img_format+' % run 1 data pattern (must use wildcard)\n\n';


    with open(output_file, 'w') as f:
        f.writelines(content[1:23])

    with open(output_file, 'a') as f:
        for i, c in enumerate(onset_tasks):
            if combineSmiss == True and 'Smiss' in c:
                continue
            else:
                f.write('cond_name\t'+c+' % condition '+str(i+1)+ ' name\n')
                f.write('ref_scan_onset\t0        % reference scan onset for condition '+str(i+1)+ '\n')
                f.write('num_ref_scan\t1        % number of reference scan for condition '+str(i+1)+'\n\n')

    with open(output_file, 'a') as f:
           f.writelines(content[24:42])

    with open(output_file, 'a') as f:
        for col in onset_tasks:
            only_int = pls_onset_df[col]
            only_int = [int(x) for x in only_int if str(x) != 'nan']
            # print(only_int)
            if combineSmiss == True:
                if 'Recog' in col:
                    prefix=col.split('_')[0]
                    suffix=col.split('_')[2]
                    newcol=prefix+'_Smiss_'+suffix
                    append_list=pls_onset_df[newcol]
                    append_list=[int(x) for x in append_list if str(x) != 'nan']
                    only_int.extend(append_list)
                    # only_int.sort()
                if 'Smiss' in col:
                    continue

            # print(only_int)
            f.write('\nevent_onsets\t')
            f.writelines(["%s " %str(item) for item in only_int])


    with open(output_file, 'a') as f:
           f.writelines(content[42:])




def CountNumOfEvents(pls_onset_df):
    tasksToInclude=[key for key,value in pls_onset_df.items()]

    # count_df=pandas.DataFrame(columns = list(pls_onset_df.columns))
    count_dict=dict()
    count=[]
    for col in tasksToInclude:
        only_int = pls_onset_df[col]
        only_int = [x for x in only_int if str(x) != 'nan']
        print(only_int)
        count.append(len(only_int))
        count_dict[col]=len(only_int)

    # count_df=count_df.append(pandas.Series(count,index=count_df.columns), ignore_index=True)
    return count_dict

def censor_onsets(onset_df, rowNum=0, censor_list=None, tasksToInclude=None):
    pls_onset_df=onset_df.iloc[rowNum:, 1:].T
    pls_onset_df.columns=pls_onset_df.iloc[0]
    pls_onset_df=pls_onset_df[2:]
    pls_onset_df = pls_onset_df.dropna(how='all')
    keep_cols=[c for c in pls_onset_df.columns if c in tasksToInclude]
    # print(keep_cols)
    pls_onset_df=pls_onset_df[keep_cols]

    pls_onset_dict_censored=dict()
    remove_list=[]
    remove_list_onset_name=[]

    for (onsetName, onsetValue) in pls_onset_df.iteritems():
        # if "".join(re.split("[^a-zA-Z]+", task)) in onsetName.lower():
        only_int = onsetValue.tolist()
        only_int = [x for x in only_int if str(x) != 'nan']

        for trial in only_int:
            for censor_vol in censor_list:
            # print(trial+8)

                if censor_vol in range(int(trial), int(trial+9)):
                    print("Onset Name:", onsetName)
                    print("Onset to be censored: ", trial)

                    if trial not in remove_list:
                        remove_list.append(trial)
                        remove_list_onset_name.append(onsetName)

        only_int=[x for x in only_int if x not in remove_list]
        #Put censored onsets in a dictionary
        h=defaultdict(list)
        for key, val in zip(remove_list_onset_name, remove_list):
            if val not in h[key]:
                h[key].append(val)

        remove_dict=dict(h)

        pls_onset_dict_censored[onsetName]=only_int


    return pls_onset_dict_censored,remove_dict
