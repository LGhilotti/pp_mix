#ifndef FIXED_COV_HPP
#define FIXED_COV_HPP

#include "base_cov.hpp"

class FixedCov: public BaseCov {
 protected:
    int dim;
    double sigma;
    bool return_chol = true;

 public:
    FixedCov(int dim, double sigma): dim(dim), sigma(sigma) {}

    ~FixedCov() {}

    CovMat sample_prior() override;

    CovMat sample_given_data(
        const std::vector<VectorXd> &data, const CovMat &curr,
        const VectorXd &mean) override;
};

class FixedPrec : public BasePrec
{
protected:
   int dim;
   double sigma;

public:
   FixedPrec(int dim, double sigma) : dim(dim), sigma(sigma) {}

   ~FixedPrec() {}

   PrecMat sample_prior() override;

   PrecMat sample_given_data(
       const std::vector<VectorXd> &data, const PrecMat &curr,
       const VectorXd &mean) override;
};

#endif 