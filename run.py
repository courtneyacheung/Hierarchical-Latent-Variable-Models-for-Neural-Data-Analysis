from data_cleaning import region_in_good_cluster, region_search, data_cleaning, plot_trajectories2L
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
import tqdm as tqdm
from pcca_methods import PCCA

ibl_cache = Path.home() / 'Downloads' / 'IBL_Cache'
ibl_cache.mkdir(exist_ok=True, parents=True)

one = ONE(base_url='https://openalyx.internationalbrainlab.org', \
          password='international', silent=True, cache_dir=ibl_cache)

ba = AllenAtlas()

#method to train vLGP model
def train_model(sessionTrain, sessionTest, ys):
    kernel = RBF(scale=1., lengthscale=0.3)#10 * dt)
    sessionTrain, params = vi.fit(sessionTrain, n_factors=2, kernel=kernel, seed=0, max_iter=50, trial_length=ys[0].shape[1])#, GPFA=True)
    z_train = rearrange(sessionTrain.z, '(trials time) lat -> trials time lat', time=ys[0].shape[1])
    # Infer latents of test data
    sessionTest = vi.infer(sessionTest, params=params)
    z_test = rearrange(sessionTest.z, '(trials time) lat -> trials time lat', time=ys[0].shape[1])
    return z_train, z_test

acronym = 'SCdg'
insertions = one.search_insertions(atlas_acronym=acronym, query_type='remote')

#method to train vLGP on a brain region and plot the learnt trajectories
def train_vLGP(region_name, insertions):
    spikes, clusters_good, spikes_g, events, trials, contrast, choice, accuracy = region_search(region_name, insertions, 32)
    sessionTrain, sessionTest, ys, num_train = data_cleaning(region_name , spikes, clusters_good, spikes_g, events, trials)
    z_train, z_test = train_model(sessionTrain, sessionTest, ys)
    plot_trajectories2L(z_train, choice[:num_train], accuracy[:num_train], 0.05 *1000, region_name = region_name)
    return z_train, z_test

#learn latent trajectories for the two regions
z_train_SCdg, z_test_SCdg = train_vLGP('SCdg', insertions)
z_train_SCiw, z_test_SCiw = train_vLGP('SCiw', insertions)

#train them on pCCA model

#reshape the latent trajectory data learnt from vLGP to fit the input type of the pCCA model
def pcca_data_cleaning(z):
   z_new = np.transpose(z, (2, 0, 1))
   z_new  = np.reshape(z_new, (z_new.shape[0], z_new.shape[1]*z_new.shape[2])).T
   return z_new

#make prediction of the latent trajectories based on the trained pCCA model
def pcca_on_multiregions_pred(n_components, X1, X2):
  pcca = PCCA(n_components=n_components)
  pcca.fit([X1, X2], n_iters=100)
  X1_, X2_ = pcca.sample(X1.shape[0])
  return X1_, X2_

#helper method to calculate RMSE
def rmse_calc(X_true, X_estimated):
  return np.sqrt(np.mean((X_estimated - X_true) ** 2))

#calculate RMSE for a pCCA model given the number of latent varibles learnt
def pcca_rmse_calc(n_components, X1, X2):
  X1_, X2_ = pcca_on_multiregions_pred(n_components=n_components, X1= X1, X2 = X2)
  return rmse_calc(X1, X1_), rmse_calc(X2, X2_), (rmse_calc(X1, X1_)+ rmse_calc(X2, X2_))

#reshape the data to fit in the pCCA model
z_region1_new = pcca_data_cleaning(z_train_SCdg)
z_region2_new = pcca_data_cleaning(z_train_SCiw)
print(z_region1_new)
print(z_region2_new)

rmse_dict = {}
for i in range(1,7):
  rmse_dict["latent"+str(i)] = [pcca_rmse_calc(n_components = i, X1= z_region1_new, X2 = z_region2_new)]
rmse_dict

# # #plot the RMSE for pCCA model with different numbers of latent vraible learnt
fig = plt.figure()
rmse_region1 = [v[0][0] for v in rmse_dict.values()]
rmse_region2 = [v[0][1] for v in rmse_dict.values()]
x_tick_lst = [1,2,3,4,5,6]
plt.plot(x_tick_lst,rmse_region1, label = 'SCdg')
plt.plot(x_tick_lst,rmse_region2, label = 'SCiw')
plt.xticks(np.arange(1, 7, step = 1))
plt.xlabel('Number of Latent Variables', fontsize = 18)
plt.ylabel('RMSE', fontsize = 18)
plt.legend(fontsize = 12)
plt.suptitle('pCCA RMSE plot', fontsize = 20)
fig.show()
fig.savefig('results/pCCA_RMSE_plot.png')
