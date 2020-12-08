import os
import glob 
import nilearn
import re
import json
import nibabel as nib
import pandas as pd
from nilearn.image import mean_img
import numpy as np
from scipy.io import loadmat

def time_to_slice(time_intervals,tr,last_event_tint):
    
    """
      Returns a list of slices indices to consider 
      for the averaging for each event.
      
    """
    
    max_time = int(np.ceil(last_event_tint[1]))
    slices_tint = [] # intervals of time for every slice
    
    # make even
    if max_time % 2 != 0:
      max_time+=1
      
    for i in range(0,max_time-2):
      l_int = tr*i
      u_int = tr*i + tr
      slices_tint.append((l_int,u_int))
    
    slice_nums = []
    for tint in time_intervals:
      lower_event_time = tint[0]
      upper_event_time = tint[1]
      
      for slice_num, j in enumerate(slices_tint):
        lower_scan_time = j[0]
        upper_scan_time = j[1]
        
        if lower_scan_time <= lower_event_time <= upper_scan_time: # beginning of event happens while scan happened
          
          if upper_event_time <= upper_scan_time: # if event is contained inside a single scan
            slice_nums.append([slice_num])
            print(f"event in time window: {(lower_event_time,upper_event_time)} is in scan interval: {(lower_scan_time,upper_scan_time)} -> slice# {slice_num}")
            
          elif upper_event_time > upper_scan_time: # event overlaps with two slices
            slice_nums.append([slice_num,slice_num+1])
            print(f"event in time window: {(lower_event_time,upper_event_time)} is in scan intervals: {((lower_scan_time,upper_scan_time),(upper_scan_time,upper_scan_time+tr))} -> slice# {(slice_num,slice_num+1)}")
        
          break # we found the correct scans, move on to the next event
    
    return slice_nums

def extract_time_series_openneuro(fmri_path,event_timing_path,raw_4d_info_path):
  
  """ 
     - all time is in ms
     - the first onset time is from the first scan
     - the duration corresponds to the time before the subject answers i.e. the subject reaction time
     since right after this, the next cue appears
     - the time interval between onsets corresponds to the TR time and is roughly constant
     - it SEEMS like 
  
  """
  
  event_name_to_int = {'go':0,'failed stop':-1, 'successful stop':1}
  
  # load fmri file 
  fmri=nib.load(fmri_path)
  data = fmri.get_fdata()
  header = fmri.header
  
  # confirm from raw file the following:
  # 1. the number of slices and 2. the TR timing
  #and get TR: http://mriquestions.com/tr-and-te.html

  # read 4d run info
  with open(raw_4d_info_path) as f:
    info = json.load(f)
  
  tr = info['RepetitionTime']

  #shape = header.get_data_shape()
  
  # load event file
  events = pd.read_csv(event_timing_path,sep='\t')
  
  # replace string event identifiers with integers
  events.trial_type = events.trial_type.replace(event_name_to_int)
    
  number_of_events = len(event_name_to_int)
  grouped_data = events.groupby('trial_type')
  
  # what will be returned
  labels = []    
  event_samples = []
  
  for event, data in  grouped_data:
    
    if event == 'junk':
      continue
    
    onsets = data['onset'].values
    durations = data['duration'].values
    last_event_tint = (onsets[-1],onsets[-1]+durations[-1])
    
    # get the interval (onset,onset+duration) for each event
    #TODO: probably want to extend this window, maybe just add 
    time_intervals = [(onsets[i], (onsets[i]+ durations[i]) ) for i in range(len(onsets))]
    
    # convert time intervals to slices numbers for each occurance of the event
    slice_intervals = time_to_slice(time_intervals,tr,last_event_tint) 
    
    # extract the relevant slices 
    #TODO: this might be where we want to downsample the files see: https://nipy.org/nibabel/nibabel_images.html#the-image-header
    for sint in slice_intervals:
      
      if len(sint) > 1: # more than one slice to consider
        imgs = fmri.slicer[...,sint[0]:sint[1]+1] 
        avg_img = mean_img(imgs) # this is a NiftiImage obj
        
      else:
        avg_img = fmri.slicer[...,sint[0]]
      
      #print(f"Appending X of shape {avg_img.shape}, and y={event}...")
      event_samples.append(avg_img.get_fdata())
      labels.append(np.array(event))
    
  
  # define correspondence between event timing in ms 
  # and slice number 
  
  print(f"Make sure this matches: num labels: {len(labels)}, num samples: {len(event_samples)}")
  return labels, event_samples




# def extract_time_series(fmri_path,event_timing_path,raw_4d_info_path):
#   
#   """ 
#      - all time is in ms
#      - the first onset time is from the first scan
#      - the duration corresponds to the time before the subject answers i.e. the subject reaction time
#      since right after this, the next cue appears
#      - the time interval between onsets corresponds to the TR time and is roughly constant
#      - it SEEMS like 
#   
#   """
#   
#   
#   # load fmri file 
#   fmri=nib.load(fmri_path)
#   data = fmri.get_fdata()
#   header = fmri.header
#   
#   # confirm from raw file the following:
#   # 1. the number of slices and 2. the TR timing
#   #and get TR: http://mriquestions.com/tr-and-te.html
# 
#   # read 4d run info
#   with open(raw_4d_info_path) as f:
#     info = json.load(f)
#   
#   TR = info['RepetitionTime']
#   #raw_4d_header = nib.load(raw_4d_file).header
#   #TR = float(re.search('TR:(\d+)',str(raw_4d_header['db_name'])).group(1))
#   #TE = float(re.search('TE:(\d+)',str(raw_4d_header['db_name'])).group(1))
#   
#   #shape = header.get_data_shape()
#   
#   # load event file
#   events = loadmat(event_timing_path)
#   number_of_events = len(events.keys())
#   event_names = []
#   durations = []
#   onsets = [] 
#   for i in  range(number_of_events):
#     event_names.append(events['names'][0][i][0])
#     durations.append(events['duration'][0][i])
#     onsets.append(events['onsets'][0][i])
#     
#     # get the interval (onset,onset+duration) for each event
#     #TODO: probably want to extend this window, maybe just add 
#     tint = [(onsets[i], (onsets[i]+ durations[i]) ) for i in range(len(onsets[i]))]
#     #event_names = [events['names'][0][i][0] for i in range(3,(3+number_of_events))]
#     #durations = [events['durations'][0][i] for i in range(3,(3+number_of_events))]
#     #onsets = [events['onsets'][0][i] for i in range(3,(3+number_of_events))]
#     
#     time_intervals = [] # list of time intervals from onset + duration 
#     
#   # convert time intervals to slices:
#   slice_intervals = [time_to_slice(tint) for tint in time_intervals]
#   
#   # extract the relevant slices 
#   #TODO: this might be where we want to downsample the files see: https://nipy.org/nibabel/nibabel_images.html#the-image-header
#   event_samples = [fmri.slicer[...,sint] for sint in slice_intervals]
#   
#   # define correspondence between event timing in ms 
#   # and slice number 
#   def time_to_slice(time,tr)
#   
#     return slice_intervals
#   
#   def read_event_mat_file(event_file_path):
#     
#     return time_intervals
#     
#   # return two arrays: y: binary response vector
#   #                    X: MxN matrix of M samples by N voxels
#   return {'y':None, 'X': None}


def collect_data(maindir,task,timepoint,outputdir,N=None):
  
  # get the fmri file and event timing file as tupe for everyone
  # fmri_files_path = os.path.join(maindir,'nii_files',task,f'V{timepoint}')
  # event_files_path = os.path.join(maindir,'mat_files',task,f'V{timepoint}','temp')
  # fmri_files = glob.glob(fmri_files_path,f'sub**','swa*{task.capitalize()}*.nii')
  fmri_files_path = os.path.join(maindir,'derivatives2/fmriprep')
  event_files_path = fmri_files_path
  
  fmri_files = glob.glob(os.path.join(fmri_files_path,f'sub**','func','sub-*_task-stopmanual_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
  
  if N != None:
    fmri_files = fmri_files[:N]
  
   
  list_of_X = []
  list_of_y = []
  
  for fmri_path in fmri_files:
    subject = os.path.dirname(fmri_path).split('/')[-2]
    
    print(f"Processing subject {subject}")
    
    event_timing_path = os.path.join(maindir,subject,'func',f'{subject}_task-stopmanual_run-1_events.tsv')
    raw_4d_info_path = os.path.join(maindir,subject,'func',f'{subject}_task-stopmanual_run-1_bold.json')
    #event_timing_path = os.path.join(event_files_path,f'{task.upper()}_NV{fmri_path[-3:]}_events.mat')
    #raw_4d_file = os.path.join('/data/neuroventure/raw/task_fmri/{task}/V{timepoint}/{subject[-3:]}{task.capitalize()}.nii')
    
    # append the data from one subject
    y, X = extract_time_series_openneuro(fmri_path,event_timing_path,raw_4d_info_path)
    list_of_X.append(X)
    list_of_y.append(y)
    
  # transform list of tuples into one response vector
  # and one matrix of size (S*2M)xN, where S= number of subjects, 
  # M=number of samples from the 4d time series of each events and neutral scans
  # N= number of voxels
  y = np.concatenate(list_of_y)
  X = np.concatenate(list_of_X)
  
  print(f"y shape: {y.shape}, X shape: {X.shape}")
  print(f"Finished subject {subject}...")
    
  
  return X, y
  
  
############ test 
print("Begin data prep: extracting time series to label...")

maindir = '/data/datasets/open-neuro/ds007_R2.0.0/'
outputdir = '/home/sean/wd/pgm-fmri/data' # relative to maindir
task='stop'
timepoint='1'

N=2 # number of subjects to include. If none include all

X, y = collect_data(maindir,task,timepoint,outputdir,N)

print("Saving X and y...")

np.save(os.path.join(outputdir,'X.npy'),X)
np.save(os.path.join(outputdir,'y.npy'),y)


print("Done.")


