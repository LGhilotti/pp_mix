#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <memory>

#include "conditional_mcmc.hpp"

#include "precs/base_prec.hpp"
#include "precs/fixed_prec.hpp"
#include "precs/delta_wishart.hpp"
#include "precs/delta_gamma.hpp"

#include "point_process/determinantalPP.hpp"

#include "../protos/cpp/params.pb.h"

// SAMPLER
MCMCsampler::MultivariateConditionalMCMC* make_sampler(const Params& params, DeterminantalPP* pp, BasePrec* g);

// Lambda sampler
MCMCsampler::BaseLambdaSampler* make_LambdaSampler(MCMCsampler::MultivariateConditionalMCMC* mcmc, const Params& params);

// AMeans sampler
MCMCsampler::BaseMeansSampler* make_MeansSampler(MCMCsampler::MultivariateConditionalMCMC* mcmc, const Params& params);

// DPP
DeterminantalPP* make_dpp(const Params& params, const MatrixXd& ranges);

// Delta Precision
BasePrec* make_delta(const Params& params);

BasePrec *make_fixed_prec(const FixedMultiPrecParams &params);

BasePrec* make_wishart(const WishartParams& params);

BasePrec* make_fixed_prec(const FixedUnivPrecParams& params);

BasePrec* make_gamma_prec(const GammaParams& params);

#endif
