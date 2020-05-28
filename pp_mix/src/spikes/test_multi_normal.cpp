#include "../precs/precmat.hpp"
#include "../precs/wishart.hpp"
#include "../utils.hpp"

#include <vector>
#include <stan/math/prim/mat.hpp>

int main() {
    int dim = 2;
    Wishart wishart_prec(dim + 3.0, dim, 1.0);
    PrecMat prec = wishart_prec.sample_prior();

    Eigen::VectorXd x = Eigen::VectorXd::Ones(dim) * 2.5;
    Eigen::VectorXd m = Eigen::VectorXd::Ones(dim);

    std::cout << "our: " << o_multi_normal_prec_lpdf(x, m, prec) + NEG_LOG_SQRT_TWO_PI * dim << std::endl;
    std::cout << "stan: " << stan::math::multi_normal_prec_lpdf(x, m, prec.get_prec()) << std::endl;
}