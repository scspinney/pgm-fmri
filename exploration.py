import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def plot_event_signal(df):
  df['time'] = (df['goscreen.OnsetTime'] - df['Trigger.RTTime']) / 1000.0
  df['event']=0
  df['event'][df['trialtype'] == 'stoptrial'] = 1
  #df['time'].plot()
  sns.lineplot(x='time',y='event',data=df)
  #plt.show()
  plt.savefig('stop_event_time_series.png')
  
  

df = pd.read_excel('Stop_V1.xlsx')

single_sub_df = df[df["Subject"]==2]
print(single_sub_df.head())
plot_event_signal(single_sub_df)
