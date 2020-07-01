#include "wishart.hpp"

Wishart::Wishart(double df, int dim, double sigma): df(df) {
    this->psi = Eigen::MatrixXd::Identity(dim, dim) * sigma;
    this->inv_psi = Eigen::MatrixXd::Identity(dim, dim) * 1.0 / sigma;
}

// Wishart::Wishart(double df, const MatrixXd &psi): df(df), psi(psi) {
//     std::cout << "Wishart::Wishart strange" << std::endl;
//     inv_psi = psi.inverse();
// }

PrecMat Wishart::sample_prior() {
    MatrixXd out = wishart_rng(df, psi, Rng::Instance().get());
    return PrecMat(out);
}

PrecMat Wishart::sample_given_data(
    const std::vector<VectorXd> &data, const PrecMat &curr,
    const VectorXd &mean)
{
    MatrixXd data_mat = vstack(data);
    data_mat = data_mat.rowwise() - mean.transpose();

    MatrixXd out = wishart_rng(
        df + data.size(),
        (inv_psi + data_mat.transpose() * data_mat).inverse(),
        Rng::Instance().get());

    return PrecMat(out);
}

PrecMat Wishart::mean() const
{
    return PrecMat(psi * df);
}