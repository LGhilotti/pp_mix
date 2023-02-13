import numpy as np
from google.protobuf import text_format

import pp_mix.protos.py.params_pb2 as params_pb2

from pp_mix.interface import ConditionalMCMC, cluster_estimate
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params

chain = loadChains("data/Bauges_data/applam_d_4_fixed_out/rho_50_ranges_50_out_0/chains.recordio", MultivariateMixtureState)

prod_col = np.empty(len(chain))
i=0
for x in chain:
    lamb_eigen = x.lamb_block.lamb
    lamb_np = to_numpy(lamb_eigen)
    lamb_lambT = np.matmul(lamb_np, lamb_np.transpose())
    prod_col[i] = lamb_lambT
    i=i+1

p = chain[0].lamb_block.lamb.rows
d = chain[0].lamb_block.lamb.cols

est = np.zeros((p,d))
for i in range(len(prod_col)):
    est = est + prod_col[i]

est = est/len(prod_col)
np.savetxt("est_lamb_lambT", est)
