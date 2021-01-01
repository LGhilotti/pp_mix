#include "factory.hpp"

// DPP
BaseDeterminantalPP* make_dpp(const Params &params, const MatrixXd& ranges) {
  if (ranges.cols()==1)
    return new UniDpp(ranges, params.dpp().N(), params.dpp().c(), params.dpp().s() );
  else
    return new MultiDpp(ranges, params.dpp().N(), params.dpp().c(), params.dpp().s() );
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
