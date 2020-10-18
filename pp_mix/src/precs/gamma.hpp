#ifndef PREC_GAMMA_HPP
#define PREC_GAMMA_HPP

#include <algorithm>
#include <stan/math/prim/mat.hpp>

#include "../rng.hpp"
#include "../utils.hpp"
#include "base_prec.hpp"

class GammaPrec : public BaseUnivPrec {
 protected:
  double alpha;
  double beta;

 public:
  GammaPrec(double alpha, double beta);

  ~GammaPrec() {}

  double sample_prior() override;

  double sample_given_data(const std::vector<double> &data, const double &curr,
                           const VectorXd &mean) override;

  double mean() const override;

  double lpdf(double val) const override {
    return stan::math::gamma_lpdf(val, alpha, beta);
  };
};

#endif