import logging
import joblib
import os
import sys
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from google.protobuf import text_format
from itertools import combinations, product
from scipy.stats import multivariate_normal, norm

import pp_mix.protos.py.params_pb2 as params_pb2
from pp_mix.protos.py.state_pb2 import (
    UnivariateMixtureState, MultivariateMixtureState)
from pp_mix.protos.py.params_pb2 import Params
from pp_mix.utils import loadChains, writeChains, to_numpy, gen_even_slices
from pp_mix.params_helper import check_params, make_params
from pp_mix.precision import PrecMat

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import pp_mix_cpp  # noqa
   
   
def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out


class ConditionalMCMC(object):
    def __init__(self, params_file="", pp_params=None, prec_params=None,
                 jump_params=None, mala_stepsize=0.1):
        if params_file != "":
            with open(params_file, 'r') as fp:
                self.params = Params()
                text_format.Parse(fp.read(), self.params)

        else:
            self.params = make_params(pp_params, prec_params, 
                                      jump_params)
        
        self.params.mala_stepsize = mala_stepsize
        self.serialized_params = self.params.SerializeToString()

    def run(self, nburn, niter, thin, data, log_every=200, bernoulli=False):
        check_params(self.params, data, bernoulli)
        if data.ndim == 1:
            self.dim = 1
        else:
            self.dim = data.shape[1]

        self._serialized_chains = pp_mix_cpp._run_pp_mix(
            nburn, niter, thin, data, self.serialized_params, bernoulli, log_every)

        if self.dim == 1:
            objType = UnivariateMixtureState
        else:
            objType = MultivariateMixtureState

        self.chains = list(map(
            lambda x: getDeserialized(x, objType), self._serialized_chains))

    def serialize_chains(self, filename):
        writeChains(self.chains, filename)

    def sample_predictive(self):
        if self.dim == 1:
            out = pp_mix_cpp._sample_predictive_univ(self._serialized_chains)
        else:
            out = pp_mix_cpp._sample_predictive_multi(
                self._serialized_chains, self.dim)
        return out


class RJMCMC(object):
    def __init__(self, params_file="", pp_params=None, prec_params=None):
        if params_file != "":
            with open(params_file, 'r') as fp:
                self.params = Params()
                text_format.Parse(fp.read(), self.params)

        else:
            self.params = make_params(pp_params, prec_params)
        
        self.serialized_params = self.params.SerializeToString()

    def run(self, nburn, niter, thin, data, log_every=200, bernoulli=False):
        check_params(self.params, data, bernoulli)
        if data.ndim == 1:
            self.dim = 1
        else:
            self.dim = data.shape[1]

        self._serialized_chains = pp_mix_cpp._run_rj(
            nburn, niter, thin, data, self.serialized_params, log_every)

        if self.dim == 1:
            objType = UnivariateMixtureState
        else:
            objType = MultivariateMixtureState

        self.chains = list(map(
            lambda x: getDeserialized(x, objType), self._serialized_chains))

    def serialize_chains(self, filename):
        writeChains(self.chains, filename)

    def sample_predictive(self):
        if self.dim == 1:
            out = pp_mix_cpp._sample_predictive_univ(self._serialized_chains)
        else:
            out = pp_mix_cpp._sample_predictive_multi(
                self._serialized_chains, self.dim)
        return out


def simulate_strauss2d(ranges, beta, gamma, R):
    return pp_mix_cpp._simulate_strauss2D(ranges, beta, gamma, R)


def estimate_multi_density(state, grid):
    dim = grid.shape[1]
    norm_ = multivariate_normal
    T = np.sum(state.a_jumps.data) + np.sum(state.na_jumps.data)
    out = np.zeros(grid.shape[0])
    for i in range(state.ma):
        prec = PrecMat(to_numpy(state.a_precs[i]))
        out += state.a_jumps.data[i] / T * np.exp(
            norm_._logpdf(grid, to_numpy(state.a_means[i]), prec.prec_cho,
                         prec.log_det_inv, dim))

    for i in range(state.mna):
        prec = PrecMat(to_numpy(state.na_precs[i]))
        out += state.na_jumps.data[i] / T * np.exp(
            norm_._logpdf(grid, to_numpy(state.na_means[i]), prec.prec_cho,
                         prec.log_det_inv, dim))

    return out


def estimate_univ_density(state, grid):
    T = np.sum(state.a_jumps.data) + np.sum(state.na_jumps.data)
    out = np.zeros(grid.shape[0])
    for i in range(state.ma):
        sd = 1.0 / np.sqrt(state.a_precs.data[i])
        out += state.a_jumps.data[i] / T * norm.pdf(
            grid, state.a_means.data[i], sd)

    for i in range(state.mna):
        sd = 1.0 / np.sqrt(state.na_precs.data[i])
        out += state.na_jumps.data[i] / T * norm.pdf(
            grid, state.na_means.data[i], sd)
    return out



def estimate_density_seq(mcmc_chains, grid):
    dim = grid.ndim
    
    if dim == 1:
        out = np.zeros((len(mcmc_chains), len(grid)))
        dens_func = estimate_univ_density
    else:
        out = np.zeros((len(mcmc_chains), grid.shape[0]))
        dens_func = estimate_multi_density

    for i, state in enumerate(mcmc_chains):
        out[i, :] = dens_func(state, grid)
    return out


def lpml(data, chains):
    densities = estimate_density_seq(chains, data) 
    cpos = 1 / np.mean(1 / densities, axis=0)
    return np.mean(np.log(cpos))


def minbinder_clus(chains, njobs=-1):
    def _loss_fun(clus, psm):
        aff_matrix = np.zeros((ndata, ndata))
        clus_vals = np.unique(clus)
        for k in clus_vals:
            for i, j in combinations(np.where(clus == k)[0], 2):
                aff_matrix[i, j] = 1
            
        aff_matrix += np.transpose(aff_matrix)
        np.fill_diagonal(aff_matrix, 1.0)
        
        return np.sum((aff_matrix - psm) ** 2) 
            

    clus_chain = np.vstack([x.clus_alloc for x in chains])
    ndata = clus_chain.shape[1]
    psm = np.zeros((ndata, ndata))
    for i in range(ndata):
        psm[i, i] = 0.5
        for j in range(i):
            psm[i, j] = np.mean(clus_chain[:, i] == clus_chain[:, j])
        
    psm += np.transpose(psm)

    if njobs < 1:
        njobs = joblib.cpu_count() + njobs
    
    fd = delayed(_loss_fun)
    losses = Parallel(n_jobs=njobs)(
        fd(clus_chain[s, :], psm)
        for s in gen_even_slices(len(chains), effective_n_jobs(njobs)))

    best = np.argmin(losses)
    return clus_chain[best, :]