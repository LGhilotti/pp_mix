#include "fixed_cov.hpp"

CovMat FixedCov::sample_prior()
{   
    Eigen::MatrixXd out = sigma * Eigen::MatrixXd::Identity(dim, dim);
    return CovMat(out);
}

CovMat FixedCov::sample_given_data(
    const std::vector<VectorXd> &data, const CovMat &curr,
    const VectorXd &mean)
{
    return sample_prior();
}

PrecMat FixedPrec::sample_prior()
{
    Eigen::MatrixXd out = sigma * Eigen::MatrixXd::Identity(dim, dim);
    return PrecMat(out);
}

PrecMat FixedPrec::sample_given_data(
    const std::vector<VectorXd> &data, const PrecMat &curr,
    const VectorXd &mean)
{
    return sample_prior();
}
