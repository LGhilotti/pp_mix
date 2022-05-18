# NOTE: TO BE LAUNCHED FROM pp_mix

#############################################
### IMPORT LIBRARIES AND FUNCTIONS ##########
#############################################

import numpy as np
import arviz as az
import os
import pandas as pd
import statistics as stat
from sklearn.decomposition import TruncatedSVD

# import pymc3 as pm
import pickle
import matplotlib.pyplot as plt
from google.protobuf import text_format
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm, mode
from scipy.interpolate import griddata
from sklearn.metrics import adjusted_rand_score
from math import sqrt

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
rho = 100.

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
ntrick =100000
nburn=100000
niter = 50000
thin= 20
log_ev=50

###################################
######## MCMC SAMPLER #############
###################################

# Build the sampler
sampler = ConditionalMCMC(hyperpar = hyperpar)

# Run the algorithm
sampler.run(ntrick, nburn, niter, thin, data_scaled, d, log_every = log_ev)

base_outpath = "data/Eyes_data/out{0}"
i = 0
while os.path.exists(base_outpath.format(i)):
    i = i+1
outpath = base_outpath.format(i)
os.makedirs(outpath)

# Save the serialized chain produced by the sampler
sampler.serialize_chains(os.path.join(outpath, "chains.recordio"))


# save the parameters
with open(os.path.join(outpath, "params.asciipb"), 'w') as fp:
    fp.write(text_format.MessageToString(hyperpar))


chain = sampler.chains

# plots
fig = plt.figure()
tau_chain = np.array([x.lamb_block.tau for x in chain])
plt.plot(tau_chain)
plt.title("tau chain")
plt.savefig(os.path.join(outpath, "tau_chain.pdf"))
plt.close()

fig = plt.figure()
first_sbar_chain = np.array([to_numpy(x.sigma_bar)[0] for x in chain])
plt.plot(first_sbar_chain,color='red')
last_sbar_chain = np.array([to_numpy(x.sigma_bar)[-1] for x in chain])
plt.plot(last_sbar_chain,color='blue')
plt.title("sbar_chain")
plt.savefig(os.path.join(outpath, "sbar_chain.pdf"))
plt.close()

# Compute Posterior Summaries
fig = plt.figure()
n_cluster_chain = np.array([x.ma for x in chain])
plt.plot(n_cluster_chain)
plt.title("number of clusters chain")
plt.savefig(os.path.join(outpath, "nclus_chain.pdf"))
plt.close()

fig = plt.figure()
n_nonall_chain = np.array([x.mna for x in chain])
plt.plot(n_nonall_chain)
plt.title("number of non allocated components chain")
plt.savefig(os.path.join(outpath, "non_alloc_chain.pdf"))
plt.close()

post_mode_nclus = mode(n_cluster_chain)[0][0] # store in dataframe
post_avg_nclus = n_cluster_chain.mean() # store in dataframe
post_avg_nonall =  n_nonall_chain.mean() # store in dataframe

clus_alloc_chain = [x.clus_alloc for x in chain]
best_clus = cluster_estimate(np.array(clus_alloc_chain))
np.savetxt(os.path.join(outpath, "best_clus.txt"), best_clus)

n_clus_best_clus = np.size(np.unique(best_clus))
true_clus = cifu_clustering.to_numpy()
ari_best_clus = adjusted_rand_score(true_clus, best_clus) # store in dataframe
aris_chain = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain])
mean_aris, sigma_aris = np.mean(aris_chain), np.std(aris_chain) # store mean_aris in dataframe
CI_aris = norm.interval(0.95, loc=mean_aris, scale=sigma_aris/sqrt(len(aris_chain))) # store in dataframe
list_performance = list()
list_performance.append([sampler.means_ar, sampler.lambda_ar, post_mode_nclus,
                    post_avg_nclus, post_avg_nonall, ari_best_clus, CI_aris])
df_performance = pd.DataFrame(list_performance, columns=('means_ar','lambda_ar',
                                      'mode_nclus', 'avg_nclus', 'avg_nonalloc', 'ari_best_clus', 'CI_aris'))
df_performance.to_csv(os.path.join(outpath, "df_performance.csv"))

theta = np.linspace(0, 2 * np.pi, data.shape[1])
dx = theta[1] - theta[0]
fig = plt.figure()

for i in range(n_clus_best_clus):
    ax = fig.add_subplot(int(sqrt(n_clus_best_clus)),int(sqrt(n_clus_best_clus)),i+1, polar=True)

    ax.plot(theta, data[best_clus == i,:].T)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.tight_layout()
plt.savefig(os.path.join(outpath, "clustered_data.pdf"))
