#include "factory.hpp"


// Lambda sampler
MCMCsampler::BaseLambdaSampler* make_LambdaSampler(MCMCsampler::MultivariateConditionalMCMC* mcmc, const Params& params){
    
    if (params.step_lambda_case()==Params::StepLambdaCase::kMhSigmaLambda)
      return new MCMCsampler::LambdaSamplerClassic(mcmc, params.mh_sigma_lambda());
    else if (params.step_lambda_case()==Params::StepLambdaCase::kMalaStepLambda)
      return new MCMCsampler::LambdaSamplerMala(mcmc, params.mala_step_lambda());
}


// AMeans sampler
MCMCsampler::BaseMeansSampler* make_MeansSampler(MCMCsampler::MultivariateConditionalMCMC* mcmc, const Params& params){
    
    if (params.step_means_case()==Params::StepMeansCase::kMhSigmaMeans)
      return new MCMCsampler::MeansSamplerClassic(mcmc, params.mh_sigma_means());
    else if (params.step_means_case()==Params::StepMeansCase::kMalaStepMeans)
      return new MCMCsampler::MeansSamplerMala(mcmc, params.mala_step_means());
}

// DPP
DeterminantalPP* make_dpp(const Params& params, const MatrixXd& ranges){

    return new DeterminantalPP(ranges, params.dpp().n(), params.dpp().c(), params.dpp().s() );
    
}


// Delta Precision
BasePrec *make_delta(const Params &params) {
  BasePrec *out;
  if (params.has_fixed_multi_prec())
    out = make_fixed_prec(params.fixed_multi_prec());
  else if (params.has_wishart())
    out = make_wishart(params.wishart());
  else if (params.has_fixed_univ_prec())
    out = make_fixed_prec(params.fixed_univ_prec());
  else if (params.has_gamma_prec())
    out = make_gamma_prec(params.gamma_prec());

  return out;
}

BasePrec *make_fixed_prec(const FixedMultiPrecParams &params) {
  return new Delta_FixedMulti(params.dim(), params.sigma());
}

BasePrec *make_wishart(const WishartParams &params) {
  params.PrintDebugString();
  double sigma = 1.0;
  if (params.sigma() > 0) {
    sigma = params.sigma();
  }
  return new Delta_Wishart(params.nu(), params.dim(), sigma);
}

BasePrec *make_fixed_prec(const FixedUnivPrecParams &params) {
  return new Delta_FixedUniv(params.sigma());
}

BasePrec *make_gamma_prec(const GammaParams &params) {
  return new Delta_Gamma(params.alpha(), params.beta());
}
