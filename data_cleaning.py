from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
# from brainbox.ephys_plots import plot_brain_regions
# from brainbox.behavior.wheel import velocity
# from brainbox.task.trials import get_event_aligned_raster, get_psth
from ibllib.atlas import AllenAtlas
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import math
# import jax.random
import numpy as np
from vlgpax.model import Session
# import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

ibl_cache = Path.home() / 'Downloads' / 'IBL_Cache'
ibl_cache.mkdir(exist_ok=True, parents=True)
one = ONE(base_url='https://openalyx.internationalbrainlab.org', \
          password='international', silent=True, cache_dir=ibl_cache)

ba = AllenAtlas()

def region_in_good_cluster(insertions, region_name):
  #STR search in good cluster
  good_region_lst = [] #list of number of the region (e.g. STR) in all good clusters of the respected insertion probe
  for i in tqdm(range(len(insertions))): #change insertions
    pid = insertions[i]
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba) #what is ba, it is brainmap
    spikes, clusters, channels = ssl.load_spike_sorting() #what is the diff btw spikes, clusters, channels?
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    good_cluster_idx = clusters['label'] >= 0.5 #to do: what does 1 mean, default idx staring from 0
    good_cluster_IDs = clusters['cluster_id'][good_cluster_idx] #label index
    # Filter the clusters accordingly:
    #clusters_g = {key: val[good_cluster_idx] for key, val in clusters.items()}
    #good_cluster_idx = clusters['label'] == 1
    clusters_good = {key:val[good_cluster_idx] for key, val in clusters.items()}
    acronyms = clusters_good['acronym'] #a list with the acronym of all clusters
    num_region = np.array([a == region_name for a in acronyms]).sum() #change this part to search for other region
    good_region_lst.append(num_region)
  return good_region_lst

def region_search(acronym, insertions, insertion_idx, label_quality=0.5, event_start_time = 'firstMovement_times'):
    # Select your PID
    pid = insertions[insertion_idx]
    #print(pid)
    [eid, pname] = one.pid2eid(pid)
    # eid = 'ebe2efe3-e8a1-451a-8947-76ef42427cc9'
    # pname = 'probe00'

    # ---------------------------------------------------
    # Load spike data
    #ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba) #what is ba, it is brainmap
    ssl = SpikeSortingLoader(one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)

    # ---------------------------------------------------
    # Restrict to only good clusters
    # Find the good cluster index:
    good_cluster_idx = clusters['label'] >= label_quality #to do: what does 1 mean, default idx staring from 0
    good_cluster_IDs = clusters['cluster_id'][good_cluster_idx] #label index
    # Filter the clusters accordingly:
    #clusters_g = {key: val[good_cluster_idx] for key, val in clusters.items()}
    clusters_good = {key:val[good_cluster_idx] for key, val in clusters.items()}
    # Filter the spikes accordingly:
    good_spk_indx = np.where(np.isin(spikes['clusters'], good_cluster_IDs))
    spikes_g = {key: val[good_spk_indx] for key, val in spikes.items()}

    # ---------------------------------------------------
    # N neuronal units in total
    num_neuron = len(np.unique(spikes_g['clusters'])) #to do: is neuronal units = individual cell

    # ---------------------------------------------------
    # Load trial data
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()
    #events = sl.trials['firstMovement_times'] #only will get movement related cell, can also ask about stimulus, session start
    #what propotion of cells are responsive to..., need to report that number
    #do not filter cells with diff function
    events = sl.trials[event_start_time] #'stimOn_times'

    # If event == NaN, remove the trial from the analysis
    nan_index = np.where(np.isnan(events))[0]
    events = events.drop(index=nan_index).to_numpy()
    contrast_R = sl.trials.contrastRight.drop(index=nan_index).to_numpy() #contrast of the stimulus appear on the right side of the screen
    contrast_L = sl.trials.contrastLeft.drop(index=nan_index).to_numpy()
    choice = sl.trials.choice.drop(index=nan_index).to_numpy() #response type: -1 turning the wheel counter clockwise, left +1 right 0 time out (didn't thurn the wheel)
    block = sl.trials.probabilityLeft.drop(index=nan_index).to_numpy() #prob the stimulus will be on the left of the screen for the current trial
    accuracy = sl.trials.feedbackType.drop(index=nan_index).to_numpy()

    # N trial count
    num_trial = len(events) #the event is when the mouse moves the trial

    # Find "trials" that go in one direction and the other direction
    # Note: This is not a pure indexing on the *task trials* as we removed trials with nan values previously
    indx_choice_a = np.where(choice == -1)[0] #turning left
    indx_choice_b = np.where(choice == 1)[0] #turning right

    contrast_R_new = np.where(np.isnan(contrast_R), 0, contrast_R)
    contrast_L_new = np.where(np.isnan(contrast_L), 0, contrast_L)
    contrast = contrast_R_new + contrast_L_new
    trials = sl.trials
    return spikes, clusters_good, spikes_g, events, trials, contrast, choice, accuracy

def data_cleaning(region_name, spikes, clusters_good, spikes_g, events, trials, bin_size = 0.05, time_bin_start = -0.1, time_bin_end = 1.05, time_window_start = 0, time_window_end = 1, random_seed = 0, subregion = True):
  #time_window = np.array([time_window_start, time_window_end])
  #events_tw = np.array([events+time_window[0], events+time_window[1]]).T
  #spike_count, cluster_id = get_spike_counts_in_bins(spikes_g['times'], spikes_g['clusters'], events_tw)
  cluster_id = np.unique(spikes_g['clusters'])
  good_cluster_df = pd.DataFrame(clusters_good)
  if subregion:
    region_df = good_cluster_df[good_cluster_df['acronym']==region_name]
    region_idx = [id for id in cluster_id if id in region_df['cluster_id'].unique()]
  else:
    region_idx = cluster_id
  spike_df = spikes.to_df()
  region_spike_df = spike_df[spike_df['clusters'].isin(region_idx)]
  cluster_spike = region_spike_df.groupby('clusters')['times'].apply(list)
  bins_scale = np.arange(time_bin_start, time_bin_end, bin_size)
  total_cluster_spike = []
  for i in range(len(trials['firstMovement_times'])): #iterate through each trial
    filter_cluster_spike = []
    for j in cluster_spike:#iterate through each cluster's spike time
      sp = (j>=trials['firstMovement_times'][i]+time_bin_start)&(j<=trials['firstMovement_times'][i]+time_bin_end)#i
      new_bin = bins_scale+trials['firstMovement_times'][i]#22bins
      bins_count = np.histogram(np.array(j)[sp],bins=new_bin)#spike split by 22 and count for each bin in that trial
      filter_cluster_spike.append(bins_count[0])
    total_cluster_spike.append(filter_cluster_spike)
  total_cluster_spike = np.array(total_cluster_spike)
  y = total_cluster_spike#spike_count_per_bin_SCdg#spike_count_per_bin_STR
  #T = total_cluster_spike.shape[1]#spike_count_per_bin_SCdg.shape[1] #5520*0.1/10 = 55.2
  dt = bin_size
  random_seed = random_seed
  num_trials = total_cluster_spike.shape[0]
  ys = y
  #session = Session(dt)  # Construct a session.

  #Create training and testing data
  num_train = int(num_trials*0.75)
  sessionTrain = Session(dt)
  for i in range(num_train):
      sessionTrain.add_trial(i, y=ys[i].T)

  sessionTest = Session(dt)
  for i in range(num_train, len(ys)):
      sessionTest.add_trial(i, y=ys[i].T)
  return sessionTrain, sessionTest, ys, num_train

#plot_trajectories2L is a function modified from the last year capstone group (Aryan Singh, Jad Makki, Saket Arora, Rishabh Viswanathan).
#This is their GitHub repository https://github.com/styyxofficial/DSC180B-Quarter-2-Project
def plot_trajectories2L(z, choices, accuracy, bin_size, region_name = 'SCdg', is_train = True):
    first = True
    first2= True
    if is_train:
        model_str = 'train'
    else:
        model_str = 'test'
    fig = go.Figure()

    for i in range(len(z)):
        if ((choices[i]==1) & (accuracy[i]==1)):
            if first:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                        mode='lines', line={'color':'blue', 'width':1}, legendgroup='right', name='Wheel Turned Right', showlegend=True))
                first = False
            else:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                            mode='lines', line={'color':'blue', 'width':1}, legendgroup='right', showlegend=False))

        elif ((choices[i]==-1) & (accuracy[i]==1)):
            if first2:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                        mode='lines', line={'color':'red', 'width':1}, legendgroup='left', name='Wheel Turned Left', showlegend=True))
                first2 = False
            else:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                            mode='lines', line={'color':'red', 'width':1}, legendgroup='left', showlegend=False))
    fig.update_layout(
        scene = dict(xaxis=dict(
            title="Time (ms)",
            title_font=dict(size=28),
            tickfont=dict(size=14)),

            yaxis=dict(
            title="Latent Variable 1",
            title_font=dict(size=28),
            tickfont=dict(size=14)),

            zaxis=dict(
            title="Latent Variable 2",
            title_font=dict(size=28),
            tickfont=dict(size=14))),
            width=1000, height=1000, title='Latent Variables over Time' )
    fig.show()
    fig.write_html('results/'+str(region_name)+'_'+model_str+'_trajectories_plot.html') 

