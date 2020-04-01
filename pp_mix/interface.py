import os
import sys
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from google.protobuf import text_format
from scipy.stats import multivariate_normal

from pp_mix.protos.py.state_pb2 import MixtureState
from pp_mix.protos.py.params_pb2 import Params
from pp_mix.utils import loadChains, writeChains, to_numpy, gen_even_slices
from pp_mix.precision.precmat import PrecMat

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import pp_mix_cpp  # noqa
   
   
def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out


class ConditionalMCMC(object):
    def __init__(self, params_file="", params=None, fixed_pp=False, inits=[]):
        self.ndim = -1

        self.fixed_pp = fixed_pp

        if params is not None:
            self.params = params

        elif params_file != "":
            with open(params_file, 'r') as fp:
                self.params = Params()
                text_format.Parse(fp.read(), self.params)

        self.serialized_params = self.params.SerializeToString()
        self.inits = inits

    def run(self, nburn, niter, thin, data):
        self._serialized_chains = pp_mix_cpp._run_pp_mix(
            nburn, niter, thin, data, self.serialized_params, 
            self.inits, self.fixed_pp)

        self.chains = list(map(
            lambda x: getDeserialized(x, MixtureState), 
            self._serialized_chains))

    def serialize_chains(self, filename):
        writeChains(self.chains, filename)


def simulate_strauss2d(ranges, beta, gamma, R):
    return pp_mix_cpp._simulate_strauss2D(ranges, beta, gamma, R)


def estimate_density(state, grid):
    dim = grid.shape[1]
    norm = multivariate_normal
    T = np.sum(state.a_jumps.data) + np.sum(state.na_jumps.data)
    out = np.zeros(grid.shape[0])
    for i in range(state.ma):
        prec = PrecMat(to_numpy(state.a_precs[i]))
        out += state.a_jumps.data[i] / T + np.exp(
            norm._logpdf(grid, to_numpy(state.a_means[i]), prec.prec_cho,
                         prec.log_det_inv, dim))

    for i in range(state.mna):
        prec = PrecMat(to_numpy(state.na_precs[i]))
        out += state.na_jumps.data[i] / T + np.exp(
            norm._logpdf(grid, to_numpy(state.na_means[i]), prec.prec_cho,
                         prec.log_det_inv, dim))

    return out


def estimate_density_seq(mcmc_chains, grid):
    out = np.zeros((len(mcmc_chains), grid.shape[0]))
    for i, state in enumerate(mcmc_chains):
        out[i, :] = estimate_density(state, grid)
    return out


def estimate_density_par(mcmc_chains, grid, njobs=-1):
    mcmc_chains = np.array(mcmc_chains)
    fd = delayed(estimate_density_seq)
    out = Parallel(n_jobs=njobs)(
        fd(mcmc_chains[s], grid.copy())
        for s in gen_even_slices(len(mcmc_chains), effective_n_jobs(njobs)))
    return np.vstack(out)

