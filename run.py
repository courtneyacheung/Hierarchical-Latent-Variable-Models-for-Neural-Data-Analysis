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
