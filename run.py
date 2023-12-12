import math

import jax.random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from vlgpax.model import Session
from vlgpax.kernel import RBF, RFF
from vlgpax import vi



random_seed = 0


def main():
    
    #when turning right
    np.random.seed(random_seed)
    # %% Generate 2D sine wave latent trajectory
    dt = 2 * math.pi * 2e-3  # stepsize
    mini_T = 500 # length 10000
    mini_t = np.arange(mini_T * dt, step=dt)  # time points
    #5000 time points each with a length of dt
    #z = np.column_stack([np.cos(0.5*t+1), np.sin(0.5*t+5)])
    z = np.column_stack([0.06*(mini_t)**2, -0.06*(mini_t-3)**2+0.54])
    # plot for right condition
    plt.plot(z)
    plt.gca().legend(('z1','z2'))
    plt.show()


    #when turning left
    np.random.seed(random_seed)
    # %% Generate 2D sine wave latent trajectory
    dt = 2 * math.pi * 2e-3  # stepsize
    mini_T = 500 # length 10000
    mini_t = np.arange(mini_T * dt, step=dt)  # time points
    #5000 time points each with a length of dt
    #z = np.column_stack([np.cos(0.5*t+1), np.sin(0.5*t+5)])
    z_2 = np.column_stack([-0.06*(mini_t-3)**2+0.54, 0.06*(mini_t)**2])
    # plot for left condition
    plt.plot(z_2)
    plt.gca().legend(('z1','z2'))
    plt.show()

    #Generate 10 trials
    right_lst = [0, 3, 5, 6, 7]

    for i in range(10):
        if i == 0:
            new_z = z
        elif i in right_lst:
            new_z = np.vstack((new_z, z))
        else:
            new_z = np.vstack((new_z, z_2))
    #plot for 10 trials
    plt.plot(new_z)

    # %% Generate Poisson observation
    T = 5000 #5000
    N = 10  # 10D number of cells
    x = np.column_stack([new_z, np.ones(T)])  # Append a constant column for bias
    #x is a matrix that contain z1, z2 columns, and a bias column
    #C is similar to A (but it is a sort of transpose of A)
    C = np.random.randn(x.shape[-1],
                        N)  # Sample the loading matrix from Gaussian
    #C has a dimension of 3(z1, z2, bias) * 10
    C[-1, :] = -1.5  # less spikes per bin
    #r is similar to big lambda n * T, our r is T * n
    r = np.exp(x @ C)  # firing rate
    y = np.random.poisson(r)  # spikes

    spike_array = y.T

    sum_lst = []
    for i in range(50):
        spk = np.sum(spike_array[:,100*i:100*(i+1)])
        #print(spike_array[:,10*i:10*(i+1)])
        sum_lst.append(spk)

    #plot PSTH
    plt.bar(np.arange(0, 5000, 100), sum_lst, width=100.0, edgecolor='black')
    plt.title('Peri-Stimulus Time Histogram (PSTH)')
    plt.xlabel('Time (100 ms)')
    plt.ylabel('Num. spike occurrences at this time')
    plt.show()

    def plot_cell(cell_number):
        rand_cell = list(y[:,cell_number-1])
        right_avg_lst = []
        left_avg_lst = []
        for i in range(10):
            if i in right_lst:
                right_avg_lst.append(rand_cell[i*500:(i+1)*500])
            else:
                left_avg_lst.append(rand_cell[i*500:(i+1)*500])

        average_right = [sum(sub_list) / len(sub_list) for sub_list in zip(*right_avg_lst)]
        average_left = [sum(sub_list) / len(sub_list) for sub_list in zip(*left_avg_lst)]

        avg_col = np.column_stack([np.array(average_right), np.array(average_left)])
        plt.plot(avg_col)
        plt.gca().legend(('right','left'))
        plt.xlabel('Time (1 ms)')
        plt.ylabel('Averge Num. spike occurrences at this time')
        plt.title('Individual cell across conditions (cell ' + str(cell_number) + ')')
        plt.show()

    #plot for cell_1
    plot_cell(1)
    #plot for cell_3
    plot_cell(3)
    #plot for cell_6
    plot_cell(6)    

    # %% Draw all
    fig, ax = plt.subplots(3, 1, sharex='all')
    ax[0].plot(new_z)  # latent
    ax[1].plot(y)  # spikes
    ax[2].imshow(y.T, aspect='auto')  # show spikes in heatmap
    
    # %% Setup inference
    ys = np.reshape(y,
                    (10, T // 10, -1))  # Split the spike train into 10 trials
    session = Session(dt)  # Construct a session.
    # Session is the top level container of data. Two arguments, binsize and unit of time, are required at construction.
    for i, y in enumerate(ys):
        session.add_trial(i + 1, y=y)  # Add trials to the session.
    # Trial is the basic unit of observation, regressor, latent factors and etc.

    # %% Build the model
    kernel = RBF(scale=1., lengthscale=100 * dt)  # RBF kernel
    # key = jax.random.PRNGKey(0)
    # kernel = RFF(key, 50, 1, scale=1., lengthscale=100 * dt)
    session, params = vi.fit(session, n_factors=2, kernel=kernel, seed=random_seed, max_iter=50)
    # `fit` requires the target `session`, the number of factors `n_factors`, and the `kernel` function.
    # `kernel` is a kernel function or a list of them corresponding to the factors.
    # RBF kernel is implemented in `gp.kernel`. You may write your own kernels.


    #10 trails turning right
    fig, ax = plt.subplots(2, 1, sharex='all')
    ax[0].plot(new_z[:500])
    ax[1].plot(session.z[:500])

    #10 trails turning left
    fig, ax = plt.subplots(2, 1, sharex='all')
    ax[0].plot(new_z[500:1000])
    ax[1].plot(session.z[500:1000])


if __name__ == '__main__':
    main()