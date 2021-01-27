import logging

import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances

import pp_mix.protos.py.params_pb2 as params_pb2
from pp_mix.protos.py.params_pb2 import Params


def check_params(params, data, ranges):
    if ranges.shape[1] != params.dimf :
        raise ValueError(
            "Ranges columns does not match factor dimension, "
            "found ranges.shape[1]={0}, dimf={1}".format(ranges.shape[1],params.dimf))
    if ranges.shape[0] != 2:
        raise ValueError(
            "Ranges should have 2 rows, incorrect number of rows, "
            "found ranges.shape[0]={0}".format(ranges.shape[0]))
                    
    if params.WhichOneof("prec_params") not in \
            ("fixed_multi_prec", "wishart"):
        raise ValueError(
            "Found {0} as precision parameter, expected one of: {1}".format(
                params.WhichOneof("prec_params"),
                "[{0}]".format(", ".join(("fixed_multi_prec", "wishart")))
            ))

    if params.prec_params.HasField("fixed_multi_prec") and params.fixed_multi_prec.dim != params.dimf :
        raise ValueError(
            "Parameter dimf should match dimension of precision matrix, "
            "found dimf={0}, prec_params.dim={1}".format(params.dimf,params.fixed_multi_prec.dim))

    if params.prec_params.HasField("wishart") and params.wishart.dim != params.dimf :
        raise ValueError(
            "Parameter dimf should match dimension of precision matrix, "
            "found dimf={0}, prec_params.dim={1}".format(params.dimf,params.wishart.dim))

                   
    if params.WhichOneof("prec_params") == "wishart":
        if params.wishart.nu < data.shape[1] + 1:
            raise ValueError(
                """Parameter wishart.nu sould be strictly greater than {0} + 1,
                 found wishart.nu={1} instead""".format(
                     data.ndim, params.wishart.nu))

    if params.wishart.identity is False:
        raise ValueError(
            "Only 'True' is supported for parametr wishart.identity")

    if params.wishart.dim != data.shape[1]:
        raise ValueError(
            "Parameter wishart.dim should match the dimension of the data, "
            "found wishart.dim={0}, data.shape[1]={1}".format(
                params.wishart.dim, data.shape[1]))

    if params.wishart.sigma <= 0:
        raise ValueError(
            "Parameter wishart.sigma should be grater than 0, "
            "found wishart.wigma={0} instead".format(params.wishart.sigma))

    if params.dpp.c <= 0 or params.dpp.s < 0 or params.dpp.s > 1:
        raise ValueError(
            "Parameter dpp are not admissible.")

    if params.a <= 0 :
        raise ValueError(
            "Parameter a (for Dirichlet) should be grater than 0, "
            "found a={0} instead".format(params.a))
    
    if params.alphajump <= 0 or params.betajump <= 0 :
        raise ValueError(
            "Gamma parameters for jumps should be greater than 0, "
            "found alphajump={0}, betajump={1}".format(params.alphajump,params.betajump))

    if params.agamma <= 0 or params.bgamma <= 0 :
        raise ValueError(
            "Gamma parameters for Sigma precision should be greater than 0, "
            "found agamma={0}, bgamma={1}".format(params.agamma,params.bgamma))

    if params.prop_means <= 0 :
        raise ValueError(
            "Parameter prop_means for MH step allocated means should be grater than 0, "
            "found prop_means={0} instead".format(params.prop_means))


    if (params.prop_lambda.HasField("MH_sigma") and params.MH_sigma <=0) or (params.prop_lambda.HasField("mala_step") and params.mala_step <=0) : 
        raise ValueError(
            "Parameter for Lambda update (MH or Mala) should be greater than 0")