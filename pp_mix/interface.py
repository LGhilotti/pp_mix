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
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector
from pp_mix.protos.py.params_pb2 import Params
from pp_mix.utils import loadChains, writeChains, to_numpy, to_proto, gen_even_slices
from pp_mix.params_helper import check_params, compute_ranges, check_ranges
from pp_mix.precision import PrecMat

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
#import pp_mix_high  # noqa


def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out


class ConditionalMCMC(object):
    def __init__(self, hyperpar):

        self.params = Params()
        self.params = hyperpar
        self.serialized_params = self.params.SerializeToString()

    def run(self, ntrick, nburn, niter, thin, data, d, ranges = 0, log_every=200):

        check_params(self.params, data, d)

        if (ranges == 0){
            ranges = compute_ranges(self.params, data, d);
        } else {
            check_ranges(ranges, d)
        }

        self.serialized_data = to_proto(data).SerializeToString()
        self.serialized_ranges = to_proto(ranges).SerializeToString()

        self._serialized_chains, self.means_ar, self.lambda_ar = pp_mix_high._run_pp_mix(
            ntrick, nburn, niter, thin, self.serialized_data, self.serialized_params, d, self.serialized_ranges, log_every)

        objType = MultivariateMixtureState

        self.chains = list(map(
            lambda x: getDeserialized(x, objType), self._serialized_chains))

    def serialize_chains(self, filename):
        writeChains(self.chains, filename)




def cluster_estimate(ca_matrix):
    serialized_ca_matrix = to_proto(ca_matrix).SerializeToString()
    serialized_best_cluster = pp_mix_high.cluster_estimate(serialized_ca_matrix)
    best_cluster = EigenVector()
    best_cluster.ParseFromString(serialized_best_cluster)
    return to_numpy(best_cluster)
