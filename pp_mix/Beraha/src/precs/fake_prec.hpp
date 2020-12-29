#ifndef FAKE_PREC
#define FAKE_PREC

#include "base_prec.hpp"

/*
 * This is a HACK! Workaround to use the ConditionalMCMC when no eta
 * parameters are present (e.g. mixtures of multivariate Bernoulli)
 */
class FakePrec: public BaseMultiPrec {
public:
    ~FakePrec() {}

    PrecMat sample_prior() { return PrecMat(); }

    PrecMat sample_given_data(const std::vector<VectorXd> &data, const PrecMat &curr,
                      const VectorXd &mean) {return PrecMat();}

    PrecMat mean() const {return PrecMat(); }

    double lpdf(const PrecMat &val) const override {return 0.0; };
};

#endif 