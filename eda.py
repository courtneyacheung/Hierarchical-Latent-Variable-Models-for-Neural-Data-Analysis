from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from ibllib.atlas import AllenAtlas

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from brainbox.task.trials import find_trial_ids
from brainbox.singlecell import bin_spikes

ibl_cache = Path.home() / 'Downloads' / 'IBL_Cache'
ibl_cache.mkdir(exist_ok=True, parents=True)

ONE.setup(silent=True)

one = ONE(base_url='https://openalyx.internationalbrainlab.org', \
          password='international', silent=True, cache_dir=ibl_cache)

ba = AllenAtlas()


import math

import jax.random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


random_seed = 0


def main():

    acronym = 'SCdg'
    insertions = one.search_insertions(atlas_acronym=acronym, query_type='remote')

    # Select your PID
    pid = insertions[32] #SCdg


    # ---------------------------------------------------
    # Convert probe PID to session EID and probe name
    [eid, pname] = one.pid2eid(pid)

    # ---------------------------------------------------
    # Load spike data
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba) #what is ba, it is brainmap
    spikes, clusters, channels = ssl.load_spike_sorting() #what is the diff btw spikes, clusters, channels?
    clusters = ssl.merge_clusters(spikes, clusters, channels)

    # ---------------------------------------------------
    # Restrict to only good clusters
    # Find the good cluster index:
    good_cluster_idx = clusters['label'] == 1 #to do: what does 1 mean, default idx staring from 0
    good_cluster_IDs = clusters['cluster_id'][good_cluster_idx] #label index
    # Filter the clusters accordingly:
    clusters_g = {key: val[good_cluster_idx] for key, val in clusters.items()}
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
    events = sl.trials['firstMovement_times'] #only will get movement related cell, can also ask about stimulus, session start
    #what propotion of cells are responsive to..., need to report that number
    #do not filter cells with diff function

    # If event == NaN, remove the trial from the analysis
    nan_index = np.where(np.isnan(events))[0]
    events = events.drop(index=nan_index).to_numpy()
    contrast_R = sl.trials.contrastRight.drop(index=nan_index).to_numpy() #contrast of the stimulus appear on the right side of the screen
    contrast_L = sl.trials.contrastLeft.drop(index=nan_index).to_numpy()
    choice = sl.trials.choice.drop(index=nan_index).to_numpy() #response type: -1 turning the wheel counter clockwise, left +1 right 0 time out (didn't thurn the wheel)
    block = sl.trials.probabilityLeft.drop(index=nan_index).to_numpy() #prob the stimulus will be on the left of the screen for the current trial

    # N trial count
    num_trial = len(events) #the event is when the mouse moves the trial

    # Find "trials" that go in one direction and the other direction
    # Note: This is not a pure indexing on the *task trials* as we removed trials with nan values previously
    indx_choice_a = np.where(choice == -1)[0] #turning left
    indx_choice_b = np.where(choice == 1)[0] #turning right

    good_cluster_idx = clusters['label'] ==1
    clusters_good = {key:val[good_cluster_idx] for key, val in clusters.items()}
    good_cluster_df = pd.DataFrame(clusters_good)

    SCdg_df = good_cluster_df[good_cluster_df['acronym']=='SCdg']
    SCiw_df = good_cluster_df[good_cluster_df['acronym']=='SCiw']
    
    def single_cluster_raster(spike_times, events, trial_idx, dividers, colors, labels, weights=None, fr=True,
                                norm=False, axs=None):

        pre_time = 0.5
        post_time = 3
        raster_bin = 0.07
        psth_bin = 0.05
        raster, t_raster = bin_spikes(
            spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=raster_bin, weights=weights)
        psth, t_psth = bin_spikes(
            spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=psth_bin, weights=weights)

        if fr:
            psth = psth / psth_bin

        if norm:
            psth = psth - np.repeat(psth[:, 0][:, np.newaxis], psth.shape[1], axis=1)
            raster = raster - np.repeat(raster[:, 0][:, np.newaxis], raster.shape[1], axis=1)

        dividers = [0] + dividers + [len(trial_idx)]
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [1, 3], 'hspace': 0}, sharex=True)
        else:
            fig = axs[0].get_figure()

        label, lidx = np.unique(labels, return_index=True)
        label_pos = []
        for lab, lid in zip(label, lidx):
            idx = np.where(np.array(labels) == lab)[0]
            for iD in range(len(idx)):
                if iD == 0:
                    t_ids = trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]
                    t_ints = dividers[idx[iD] + 1] - dividers[idx[iD]]
                else:
                    t_ids = np.r_[t_ids, trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]]
                    t_ints = np.r_[t_ints, dividers[idx[iD] + 1] - dividers[idx[iD]]]

            psth_div = np.nanmean(psth[t_ids], axis=0)
            std_div = np.nanstd(psth[t_ids], axis=0) / np.sqrt(len(t_ids))

            axs[0].fill_between(t_psth, psth_div - std_div, psth_div + std_div, alpha=0.4, color=colors[lid])
            axs[0].plot(t_psth, psth_div, alpha=1, color=colors[lid])

            lab_max = idx[np.argmax(t_ints)]
            label_pos.append((dividers[lab_max + 1] - dividers[lab_max]) / 2 + dividers[lab_max])

        axs[1].imshow(raster[trial_idx], cmap='binary', origin='lower',
                        extent=[np.min(t_raster), np.max(t_raster), 0, len(trial_idx)], aspect='auto')

        width = raster_bin * 4
        for iD in range(len(dividers) - 1):
            axs[1].fill_between([post_time + raster_bin / 2, post_time + raster_bin / 2 + width],
                                [dividers[iD + 1], dividers[iD + 1]], [dividers[iD], dividers[iD]], color=colors[iD])

        axs[1].set_xlim([-1 * pre_time, post_time + raster_bin / 2 + width])
        secax = axs[1].secondary_yaxis('right')

        secax.set_yticks(label_pos)
        secax.set_yticklabels(label, rotation=90, rotation_mode='anchor', ha='center')
        for ic, c in enumerate(np.array(colors)[lidx]):
            secax.get_yticklabels()[ic].set_color(c)

        axs[0].axvline(0, *axs[0].get_ylim(), c='k', ls='--', zorder=10)  # TODO this doesn't always work
        axs[1].axvline(0, *axs[1].get_ylim(), c='k', ls='--', zorder=10)

        return fig, axs

    def set_axis_style(ax, fontsize=12, **kwargs):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel(kwargs.get('xlabel', None), fontsize=fontsize)
        ax.set_ylabel(kwargs.get('ylabel', None), fontsize=fontsize)
        ax.set_title(kwargs.get('title', None), fontsize=fontsize)

        return ax
    

    # only for SCdg region, firstMovement_times, directions
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from First Move (s)'
        label='T from First Move (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        trial_idx, dividers = find_trial_ids(sl.trials, sort='side', order=order)
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['firstMovement_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
       
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_raster = f'results/first_movement_decision_SCdg_{ids}.png'
        
        # Save each figure with the corresponding filename
        fig.savefig(filename_raster, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)
        


    # only for SCiw, firstMovement_times, directions
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from First Move (s)'
        label='T from First Move (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        trial_idx, dividers = find_trial_ids(sl.trials, sort='side', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['firstMovement_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
    
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_raster = f'results/first_movement_decision_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_raster, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCdg region, firstMovement_times, correctness 
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from First Move (s)'
        label='T from First Move (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        
        trial_idx2, dividers2 = find_trial_ids(sl.trials, sort='choice', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['firstMovement_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)
    
        filename_corr = f'results/first_movement_correctness_SCdg_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_corr, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCiw region, firstMovement_times, correctness 
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from First Move (s)'
        label='T from First Move (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        
        trial_idx2, dividers2 = find_trial_ids(sl.trials, sort='choice', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['firstMovement_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)
    
        filename_corr = f'results/first_movement_correctness_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_corr, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCdg region, firstMovement_times, contrasts
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from First Move (s)'
        label='T from First Move (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'
        
        contrasts = np.nanmean(np.c_[sl.trials.contrastLeft, sl.trials.contrastRight], axis=1)
        trial_idx3 = np.argsort(contrasts)
        dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
        fig, axs = single_cluster_raster(spikes.times[spikes_idx], sl.trials['firstMovement_times'], trial_idx3, dividers3, colors, labels)

        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_contrasts = f'results/first_movement_contrasts_SCdg_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_contrasts, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCiw region, firstMovement_times, contrasts
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from First Move (s)'
        label='T from First Move (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'
        
        contrasts = np.nanmean(np.c_[sl.trials.contrastLeft, sl.trials.contrastRight], axis=1)
        trial_idx3 = np.argsort(contrasts)
        dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
        fig, axs = single_cluster_raster(spikes.times[spikes_idx], sl.trials['firstMovement_times'], trial_idx3, dividers3, colors, labels)

        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_contrasts = f'results/first_movement_contrasts_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_contrasts, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    ########
    #StimOn#
    ########

    # only for SCdg region, stimOn_times, directions
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Stim On (s)'
        label='T from Stim On (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        trial_idx, dividers = find_trial_ids(sl.trials, sort='side', order=order)
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['stimOn_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
       
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_raster = f'results/stimOn_times_decision_SCdg_{ids}.png'
        
        # Save each figure with the corresponding filename
        fig.savefig(filename_raster, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)
        


    # only for SCiw, stimOn_times, directions
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Stim On(s)'
        label='T from Stim On (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        trial_idx, dividers = find_trial_ids(sl.trials, sort='side', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['stimOn_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
    
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_raster = f'results/stimOn_times_decision_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_raster, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCdg region, stimOn_times, correctness 
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Stim On(s)'
        label='T from Stim On (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        
        trial_idx2, dividers2 = find_trial_ids(sl.trials, sort='choice', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['stimOn_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)
    
        filename_corr = f'results/stimOn_times_correctness_SCdg_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_corr, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCiw region, stimOn_times, correctness 
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Stim On(s)'
        label='T from Stim On (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        
        trial_idx2, dividers2 = find_trial_ids(sl.trials, sort='choice', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['stimOn_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)
    
        filename_corr = f'results/stimOn_times_correctness_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_corr, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCdg region, stimOn_times, contrasts
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Stim On(s)'
        label='T from Stim On (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'
        
        contrasts = np.nanmean(np.c_[sl.trials.contrastLeft, sl.trials.contrastRight], axis=1)
        trial_idx3 = np.argsort(contrasts)
        dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
        fig, axs = single_cluster_raster(spikes.times[spikes_idx], sl.trials['stimOn_times'], trial_idx3, dividers3, colors, labels)

        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_contrasts = f'results/stimOn_times_contrasts_SCdg_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_contrasts, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCiw region, stimOn_times, contrasts
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Stim On(s)'
        label='T from Stim On (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'
        
        contrasts = np.nanmean(np.c_[sl.trials.contrastLeft, sl.trials.contrastRight], axis=1)
        trial_idx3 = np.argsort(contrasts)
        dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
        fig, axs = single_cluster_raster(spikes.times[spikes_idx], sl.trials['stimOn_times'], trial_idx3, dividers3, colors, labels)

        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_contrasts = f'results/stimOn_times_contrasts_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_contrasts, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)


    ##########
    #Feedback#
    ##########

    # only for SCdg region, feedback_times, directions
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Feedback(s)'
        label='T from Feedback (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        trial_idx, dividers = find_trial_ids(sl.trials, sort='side', order=order)
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['feedback_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
       
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_raster = f'results/feedback_times_decision_SCdg_{ids}.png'
        
        # Save each figure with the corresponding filename
        fig.savefig(filename_raster, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)
        


    # only for SCiw, feedback_times, directions
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Feedback(s)'
        label='T from Feedback (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        trial_idx, dividers = find_trial_ids(sl.trials, sort='side', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['feedback_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
    
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_raster = f'results/feedback_times_decision_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_raster, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCdg region, feedback_times, correctness 
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Feedback(s)'
        label='T from Feedback (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        
        trial_idx2, dividers2 = find_trial_ids(sl.trials, sort='choice', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['feedback_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)
    
        filename_corr = f'results/feedback_times_correctness_SCdg_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_corr, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCiw region, feedback_times, correctness 
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Feedback(s)'
        label='T from Feedback (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'

        
        trial_idx2, dividers2 = find_trial_ids(sl.trials, sort='choice', order=order)
        
        fig, axs = single_cluster_raster(
                    spikes.times[spikes_idx], sl.trials['feedback_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])
        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)
    
        filename_corr = f'results/feedback_times_correctness_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_corr, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCdg region, feedback_times, contrasts
    for ids in np.array(SCdg_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Feedback(s)'
        label='T from Feedback (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'
        
        contrasts = np.nanmean(np.c_[sl.trials.contrastLeft, sl.trials.contrastRight], axis=1)
        trial_idx3 = np.argsort(contrasts)
        dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
        fig, axs = single_cluster_raster(spikes.times[spikes_idx], sl.trials['feedback_times'], trial_idx3, dividers3, colors, labels)

        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_contrasts = f'results/feedback_times_contrasts_SCdg_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_contrasts, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)

    # only for SCiw region, feedback_times, contrasts
    for ids in np.array(SCiw_df['cluster_id'].index):

        spikes_idx = spikes['clusters'] == ids
        order='trial num'

        xlabel='T from Feedback(s)'
        label='T from Feedback (s)'
        ylabel0= 'Firing Rate (Hz)'
        ylabel1='Sorted Trial Number'
        
        contrasts = np.nanmean(np.c_[sl.trials.contrastLeft, sl.trials.contrastRight], axis=1)
        trial_idx3 = np.argsort(contrasts)
        dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
        fig, axs = single_cluster_raster(spikes.times[spikes_idx], sl.trials['feedback_times'], trial_idx3, dividers3, colors, labels)

        set_axis_style(axs[0], ylabel=ylabel0, fontsize=16)
        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1, fontsize=16)
        fig.suptitle("unit #" + str(ids), fontsize=16)

        filename_contrasts = f'results/feedback_times_contrasts_SCiw_{ids}.png'
            
        # Save each figure with the corresponding filename
        fig.savefig(filename_contrasts, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig)


if __name__ == '__main__':
    main()