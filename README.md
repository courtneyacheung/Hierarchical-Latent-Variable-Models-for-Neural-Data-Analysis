# Hierarchical-Latent-Variable-Models-for-Neural-Data-Analysis

This project focuses on understanding neural coding, where the brain encodes sensory stimuli to generate neural and behavioral responses. Decoded through machine learning models that consider temporal correlations and non-negative support. The project uses variational Gaussian process factor analysis (vGPFA) on mouse decision-making spike train data, assuming poisson-distributed spike counts, and utilizes Bayesian inference to address non-conjugate priors in the latent space. Additionally, Canonical Correlation Analysis (CCA) and Probabilistic CCA are used to analyze neural activity correlations between the Superior Colliculus and Motor Cortex brain regions during decision-making tasks.


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`

* To get the data and trajectories results from models, run `python run.py`
  - This fetches the data, cleans data, run models and saves the result in results directory.


## Reference
Gunderson, Gregory. 2018a. “Canonical Correlation Analysis in Detail.” https://gregorygundersen.com/blog/2018/07/17/cca/

Gunderson, Gregory. 2018b. “Probabilistic Canonical Correlation Analysis in Detail.” https://gregorygundersen.com/blog/2018/09/10/pcca/#klami2015group

Keeley, Stephen, David Zoltowski, Yiyi Yu, Spencer Smith, and Jonathan Pillow. 2020.
“Efficient non-conjugate Gaussian process factor models for spike count data using polynomial approximations.” In International Conference on Machine Learning. PMLR

Rey, Hernan Gonzalo, Carlos Pedreira, and Rodrigo Quian Quiroga. 2015. “Past, present
and future of spike sorting techniques.” Brain research bulletin 119: 106–117

Lab, International Brain, Brandon Benson, Julius Benson, Daniel Birman, Niccolo
Bonacchi, Matteo Carandini, Joana A Catarino, Gaelle A Chapuis, Anne K Church-
land, Yang Dan et al. 2023. “A Brain-Wide Map of Neural Activity during Complex Be-
haviour.” bioRxiv: 2023–07

Rubin, Donald B, and Dorothy T Thayer. 1982. “EM algorithms for ML factor analysis.”
Psychometrika 47: 69–76

Yu, Byron M, John P Cunningham, Gopal Santhanam, Stephen Ryu, Krishna V Shenoy,
and Maneesh Sahani. 2008. “Gaussian-process factor analysis for low-dimensional
single-trial analysis of neural population activity.” Advances in neural information processing systems 21

Zhao, Yuan, and Il Memming Park. 2017. “Variational latent gaussian process for re-
covering single-trial dynamics from population spike trains.” Neural computation 29(5):1293–131611

Singh, Aryan, Jad Makki, Saket Arora, and Rishabh Viswanathan. 2023. "Using Latent Variable Models to Predict Mouse Behavior." https://styyxofficial.github.io/DSC180B-Quarter-2-Project/



