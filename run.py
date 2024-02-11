from data_cleaning import region_in_good_cluster, region_search, data_cleaning
import matplotlib.pyplot as plt
from vlgpax.kernel import RBF, RFF
from vlgpax import vi
from einops import rearrange
from pathlib import Path
from one.api import ONE
from ibllib.atlas import AllenAtlas
import plotly.graph_objects as go
import numpy as np
from brainbox.task.trials import find_trial_ids
from brainbox.singlecell import bin_spikes

ibl_cache = Path.home() / 'Downloads' / 'IBL_Cache'
ibl_cache.mkdir(exist_ok=True, parents=True)

one = ONE(base_url='https://openalyx.internationalbrainlab.org', \
          password='international', silent=True, cache_dir=ibl_cache)

ba = AllenAtlas()

#This is a function from the last year capstone group (Aryan Singh, Jad Makki, Saket Arora, Rishabh Viswanathan).
#This is their GitHUb repository https://github.com/styyxofficial/DSC180B-Quarter-2-Project
#We may create our onw function for plotting trajectories in future
def plot_trajectories2L(z, choices, accuracy, bin_size):
    first = True
    first2= True
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
    fig.update_layout(scene = dict(
                    xaxis_title='Time (ms)',
                    yaxis_title='Latent Variable 1',
                    zaxis_title='Latent Variable 2'),
                    width=1000, height=1000, title='Latent Variables over Time'
                    )
    fig.show()
    fig.write_html('results/SCdg_trajectories_plot.html')

def train_model(sessionTrain, sessionTest, ys):
  kernel = RBF(scale=1., lengthscale=0.3)#10 * dt)
  sessionTrain, params = vi.fit(sessionTrain, n_factors=2, kernel=kernel, seed=0, max_iter=50, trial_length=ys[0].shape[1])#, GPFA=True)
  z_train = rearrange(sessionTrain.z, '(trials time) lat -> trials time lat', time=ys[0].shape[1])

  # Infer latents of test data
  sessionTest = vi.infer(sessionTest, params=params)
  z_test = rearrange(sessionTest.z, '(trials time) lat -> trials time lat', time=ys[0].shape[1])
  return z_train, z_test

acronym_7 = 'SCdg'
insertions_7 = one.search_insertions(atlas_acronym=acronym_7, query_type='remote')

spikes, clusters_good, spikes_g, events, trials, contrast, choice, accuracy = region_search('SCdg', insertions_7, 25)
sessionTrain, sessionTest, ys, num_train = data_cleaning('SCdg', spikes, clusters_good, spikes_g, events, trials)
z_train, z_test = train_model(sessionTrain, sessionTest, ys)
plot_trajectories2L(z_train, choice[:num_train], accuracy[:num_train], 0.05 *1000)


import math

import jax.random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from src.model import Session
from src.kernel import RBF, RFF
from src import vi



random_seed = 0


def main():
    
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
    

    
    #for all region, firstMovement_times
    order='trial num'

    xlabel='T from First Move (s)'
    label='T from First Move (s)'
    ylabel0='Firing Rate (Hz)'
    ylabel1='Sorted Trial Number'

    trial_idx, dividers = find_trial_ids(trials, sort='side', order=order)
    trial_idx2, dividers2 = find_trial_ids(trials, sort='choice', order=order)
    # single_cluster_raster(spike_times, events, trial_idx, dividers, colors, labels, weights=None, fr=True,
    #                               norm=False, axs=None):
    fig, axs = single_cluster_raster(
                spikes.times, trials['firstMovement_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
    fig2, axs2 = single_cluster_raster(
                spikes.times, trials['firstMovement_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])

    #sorted by visual stimulus contrasts (0, 6.25, 12.5, 25, 100%, from pale to dark gray) and aligned to the visual stimulus onset
    contrasts = np.nanmean(np.c_[trials.contrastLeft, trials.contrastRight], axis=1)
    trial_idx3 = np.argsort(contrasts)
    dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
    labels = [str(_ * 100) for _ in np.unique(contrasts)]
    colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
    fig3, axs3 = single_cluster_raster(spikes.times, trials['firstMovement_times'], trial_idx3, dividers3, colors, labels)

    set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
    set_axis_style(axs[0], ylabel=ylabel0)

    set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
    set_axis_style(axs[0], ylabel=ylabel0)
    fig.write_html('results/first_movement_decision.html')
    fig2.write_html('results/first_movement_feedback.html')
    fig3.write_html('results/first_movement_contrast.html')


    # all region, goCue_times: The start time of the go cue tone. This is the time the sound is actually played, that is, the command sent through soundcard sync was fed back into Bpod.
    #for all region, stimOn_times
    order='trial num'

    xlabel='T from Cue On(s)'
    label='T from Cue On (s)'
    ylabel0='Firing Rate (Hz)'
    ylabel1='Sorted Trial Number'

    trial_idx, dividers = find_trial_ids(trials, sort='side', order=order)
    trial_idx2, dividers2 = find_trial_ids(trials, sort='choice', order=order)
    # single_cluster_raster(spike_times, events, trial_idx, dividers, colors, labels, weights=None, fr=True,
    #                               norm=False, axs=None):
    fig, axs = single_cluster_raster(
                spikes.times, trials['goCue_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
    fig2, axs2 = single_cluster_raster(
                spikes.times, trials['goCue_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])

    contrasts = np.nanmean(np.c_[trials.contrastLeft, trials.contrastRight], axis=1)
    trial_idx3 = np.argsort(contrasts)
    dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
    labels = [str(_ * 100) for _ in np.unique(contrasts)]
    colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
    fig3, axs3 = single_cluster_raster(spikes.times, trials['goCue_times'], trial_idx3, dividers3, colors, labels)

    set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
    set_axis_style(axs[0], ylabel=ylabel0)
    fig.write_html('results/stim_on_decision.html')
    fig2.write_html('results/stim_on_feedback.html')
    fig3.write_html('results/stim_on_contrast.html')

    # all region, feedback_times
    order='trial num'

    xlabel='T from Feedback(s)'
    label='T from Feedback (s)'
    ylabel0='Firing Rate (Hz)'
    ylabel1='Sorted Trial Number'

    trial_idx, dividers = find_trial_ids(trials, sort='side', order=order)
    trial_idx2, dividers2 = find_trial_ids(trials, sort='choice', order=order)
    # single_cluster_raster(spike_times, events, trial_idx, dividers, colors, labels, weights=None, fr=True,
    #                               norm=False, axs=None):
    fig, axs = single_cluster_raster(
                spikes.times, trials['feedback_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
    fig2, axs2 = single_cluster_raster(
                spikes.times, trials['feedback_times'], trial_idx2, dividers2, ['b', 'r'], ['correct', 'incorrect'])

    contrasts = np.nanmean(np.c_[trials.contrastLeft, trials.contrastRight], axis=1)
    trial_idx3 = np.argsort(contrasts)
    dividers3 = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
    labels = [str(_ * 100) for _ in np.unique(contrasts)]
    colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
    fig3, axs3 = single_cluster_raster(spikes.times, trials['feedback_times'], trial_idx3, dividers3, colors, labels)


    set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
    set_axis_style(axs[0], ylabel=ylabel0)
    fig.write_html('results/feedback_time_decision.html')
    fig2.write_html('results/feedback_time_feedback.html')
    fig3.write_html('results/feedback_time_contrast.html')

if __name__ == '__main__':
    main()
