# Hierarchical-Latent-Variable-Models-for-Neural-Data-Analysis

This project builds upon established dimensional reduction techniques and Bayesian inference in order to create machine learning models for decoding that account for both temporal correlations and the non-negative support of neural activity data. Specifically, we aim to learn latent neural trajectories (lower dimensional systems underlying observed neural behavior) by applying variational Gaussian process factor analysis (vLGP) to spike train data recorded from studies on decision-making in mice. This modification of GPFA assumes that spike counts are poisson-distributed, and consequently utilizes variational inference to address the issue of non-conjugate priors in the latent space by maximizing the evidence-based lower bound (ELBo). Furthermore, we analyze the correlation between neural activity in two brain regions -- the Superior Colliculus Deep Gray Layer and the Superior Colliculus Intermediate White Layer -- during a decision making task, using Probabilistic Canonical Correlation Analysis (pCCA).

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

International Brain Lab, Brandon Benson, Julius Benson, Daniel Birman, Niccolo
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



