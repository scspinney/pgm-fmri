import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import numpy as np 
from nilearn.masking import apply_mask


def plot_event_signal(df):
  df['time'] = (df['goscreen.OnsetTime'] - df['Trigger.RTTime']) / 1000.0
  df['event']=0
  df['event'][df['trialtype'] == 'stoptrial'] = 1
  #df['time'].plot()
  sns.lineplot(x='time',y='event',data=df)
  #plt.show()
  plt.savefig('stop_event_time_series.png')
  
  
def plot_3d_sample(X_path,y_path,outname):
  
  
  X = np.load(X_path)
  y = np.load(y_path)
  
  # go 
  go_ind = np.where(y==-1)[0]

  # stop fail
  sf_ind = np.where(y==1)[0]
  
  X_go = X[go_ind]
  X_sf = X[sf_ind]
  
  X_go_avg = X_go.mean(axis=0)
  X_sf_avg = X_sf.mean(axis=0)
  
  #diff = X_go_avg-X_sf_avg

  img_go = new_img_like(data=X_go_avg, ref_niimg="/data/datasets/open-neuro/ds007_R2.0.0/derivatives2/fmriprep/sub-01/func/sub-01_task-stopmanual_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz")
  img_sf = new_img_like(data=X_sf_avg, ref_niimg="/data/datasets/open-neuro/ds007_R2.0.0/derivatives2/fmriprep/sub-01/func/sub-01_task-stopmanual_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz")
  
  #img_diff = new_img_like(data=diff, ref_niimg="/data/datasets/open-neuro/ds007_R2.0.0/derivatives2/fmriprep/sub-01/func/sub-01_task-stopmanual_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz")
  #img_diff = apply_mask(img_diff, '/data/datasets/open-neuro/ds007_R2.0.0/derivatives2/fmriprep/sub-01/func/sub-01_task-stopmanual_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
  
  fig, ax = plt.subplots(nrows=1,ncols=2)
  plotting.plot_glass_brain(img_go, threshold=3, title="StopFail", axes=ax[0])
  plotting.plot_glass_brain(img_sf, threshold=3, title="StopSucc",axes=ax[1])
  plt.savefig(outname)
  plt.clf()
  
  # plotting.plot_glass_brain(img_diff,title="StopFail-StopSucc")
  # plt.savefig('sf-ss_fail.png')
  #plt.show()
  

if __name__ == "__main__":
  
  # df = pd.read_excel('Stop_V1.xlsx')
  # 
  # single_sub_df = df[df["Subject"]==2]
  # print(single_sub_df.head())
  # plot_event_signal(single_sub_df)
  plot_3d_sample('data/X.npy','data/y.npy','sample_example.png')
