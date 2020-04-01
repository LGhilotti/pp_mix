#ifndef INV_WISHART_HPP
#define INV_WISHART_HPP

#include <vector>
#include <Eigen/Dense>
#include <stan/math/prim/mat.hpp>

#include "base_cov.hpp"
#include "../rng.hpp"
#include "../utils.hpp"

using namespace Eigen;
using namespace stan::math;


class InvWishart: public BaseCov {
 protected:
    double df;
    MatrixXd psi;

 public:
     InvWishart(double df, int dim);

     InvWishart(double df, const MatrixXd& psi);

     ~InvWishart() {}

     CovMat sample_prior() override;

     CovMat sample_given_data(
         const std::vector<VectorXd> &data, const CovMat &curr,
         const VectorXd &mean) override;
};

class Wishart : public BasePrec
{
protected:
    double df;
    MatrixXd psi;
    MatrixXd inv_psi;

public:
    Wishart(double df, int dim, double sigma=1.0);

    Wishart(double df, const MatrixXd &psi);

    ~Wishart() {}

    PrecMat sample_prior() override;

    PrecMat sample_given_data(
        const std::vector<VectorXd> &data, const PrecMat &curr,
        const VectorXd &mean) override;
};

#endif 