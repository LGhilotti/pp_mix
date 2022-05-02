# NOTE: TO BE LAUNCHED FROM pp_mix

#############################################
### IMPORT LIBRARIES AND FUNCTIONS ##########
#############################################

import numpy as np
import arviz as az
import pandas as pd
import statistics as stat
from sklearn.decomposition import TruncatedSVD

# import pymc3 as pm
import pickle
import matplotlib.pyplot as plt
from google.protobuf import text_format
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.interpolate import griddata
import sys
sys.path.append('.')
sys.path.append('./pp_mix')

from pp_mix.interface import ConditionalMCMC, cluster_estimate
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params

#######################################
### READ DATA AND PRE-PROCESSING ######
#######################################

np.random.seed(1234)

# read the dataset
df = pd.read_excel("data/Eyes_data/41598_2021_2025_MOESM2_ESM.xlsx")

# extract only right eyes (OD)
df_od = df.loc[df['Masked_Id_Age1'].str.contains("OD", case=True)].set_index('Masked_Id_Age1')

# clustering obtained by cifu (method used in the paper)
cifu_clustering = df_od['Cluster_Id_Age1']

# dataset used by the algorithm, converted in numpy array
df_od = df_od[df_od.columns[1:]]
data = df_od.to_numpy()

# scaling of data
centering_var=stat.median(np.mean(data,0))
scaling_var=stat.median(np.std(data,0))
data_scaled=(data-centering_var)/scaling_var

# SVD decomposition to estimate the number of latent factors d (following Chandra)
svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
svd.fit(data_scaled)

# d is set to be the minimum number of eigenbalues explaining at least 80% of the variability in the data.
cum_eigs= np.cumsum(svd.singular_values_)/svd.singular_values_.sum()
d=np.min(np.where(cum_eigs>.80))
print("d= ",d)

####################################
##### HYPERPARAMETERS ##############
####################################

# Set hyperparameters (agreeing with Chandra)
params_file = "data/Eyes_data/resources/sampler_params.asciipb"

# Set the expected number of centers a priori
rho = 6.

# Fix "s", then: rho_max = rho/s
# It follows: c = rho_max * (2 pi)^{d/2}
s = 0.5
rho_max = rho/s
c = rho_max * ((2. * np.pi) ** (float(d)/2))

# Set the truncation level N (here called n)
n = 4

# Fill in the just computed hyperparameters in the Params object
hyperpar = Params()
with open(params_file, 'r') as fp:
    text_format.Parse(fp.read(), hyperpar)

hyperpar.dpp.c = c
hyperpar.dpp.n = n
hyperpar.dpp.s = s

print(hyperpar)

# Set sampler parameters
ntrick =10000
nburn=100000
niter = 10000
thin= 10
log_ev=50

###################################
######## MCMC SAMPLER #############
###################################

# Build the sampler
sampler = ConditionalMCMC(hyperpar = hyperpar)

# Run the algorithm
sampler.run(ntrick, nburn, niter, thin, data, d, log_every = log_ev)

# Save the serialized chain produced by the sampler
sampler.serialize_chains("data/Eyes_data/chains/chain_2.recordio")
