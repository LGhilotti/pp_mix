#ifndef INV_WISHART_HPP
#define INV_WISHART_HPP

#include <vector>
#include <Eigen/Dense>
#include <stan/math/prim/mat.hpp>

#include "base_prec.hpp"
#include "../rng.hpp"
#include "../utils.hpp"

using namespace Eigen;
using namespace stan::math;


class Wishart : public BaseMultiPrec
{
protected:
    double df;
    MatrixXd psi;
    MatrixXd inv_psi;

public:
    Wishart(double df, int dim, double sigma);

    // Wishart(double df, const MatrixXd &psi);

    ~Wishart() {}

    PrecMat sample_prior() override;

    PrecMat sample_given_data(
        const std::vector<VectorXd> &data, const PrecMat &curr,
        const VectorXd &mean) override;

    PrecMat mean() const override;

    double get_df() const {return df;}

    MatrixXd get_psi() const {return psi;}
};

#endif 