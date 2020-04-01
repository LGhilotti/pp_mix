#ifndef BASE_COV_HPP
#define BASE_COV_HPP

#include <vector>
#include <Eigen/Dense>

#include "covmat.hpp"

using namespace Eigen;


class BaseCov {
 public:
    virtual ~BaseCov() {};

    virtual CovMat sample_prior() = 0;

    virtual CovMat sample_given_data(
        const std::vector<VectorXd> &data, const CovMat &curr,
        const VectorXd &mean) = 0;
};

class BasePrec {
public:
    virtual ~BasePrec(){};

    virtual PrecMat sample_prior() = 0;

    virtual PrecMat sample_given_data(
        const std::vector<VectorXd> &data, const PrecMat &curr,
        const VectorXd &mean) = 0;
};

#endif