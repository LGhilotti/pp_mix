#include "determinantal_pp.hpp"

#include <numeric>

using stan::math::LOG_SQRT_PI;
double PI = stan::math::pi();

// initialize Kappas, phis, phi_tildes, Ds, c*, A and b
void DeterminantalPP::initialize() {
  //Sets Kappas, phis, phi_tildes and Ds
  eigen_decomposition();
  c_star = phi_tildes.sum(); // assuming B=[-1/2 , 1/2]^d

  A = MatrixXd::Zero(dim, dim);
  b = VectorXd::Zero(dim);
  for (int i = 0; i < dim; i++) {
    A(i, i) = 1.0 / (ranges(1, i) - ranges(0, i));
    b(i) = -A(i, i) * (ranges(1, i) + ranges(0, i)) / 2.0;
  }
}

double DeterminantalPP::dens(const MatrixXd &x, bool log) {
  double out;
  int n;

  // check if it's jut one point
  if ((x.size() == 1 && dim == 1) || (x.rows() == 1 & dim > 1) ||
      (x.cols() == 1 && dim > 1)) {
    n = 1;
    out =
        -1.0 * n * std::log(vol_range) + vol_range + std::log(phi_tildes.sum());

  } else {
    int n = x.rows();
    bool check_range = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        if (x(i, j) < ranges(0, j) || x(i, j) > ranges(1, j))
          check_range = false;
      }
    }
    if (check_range) {
      out = -1.0 * n * std::log(vol_range) + vol_range;

      // Transform data points to be in the unit cube centered in 0
      MatrixXd xtrans(x.rows(), x.cols());

      for (int i = 0; i < x.rows(); i++)
        xtrans.row(i) = (A * x.row(i).transpose() + b).transpose();

      // std::cout << "xtrans " << xtrans.transpose() << std::endl;
      out += log_det_Ctilde(xtrans) - Ds;
    } else {
      out = stan::math::NEGATIVE_INFTY;
    }
  }
  if (!log) out = std::exp(out);

  // std::cout << "dens: " << out << std::endl;

  return out;
}

double DeterminantalPP::papangelou(MatrixXd xi, const MatrixXd &x, bool log) {
  if (xi.cols() != dim) xi.transposeInPlace();

  MatrixXd all(xi.rows() + x.rows(), x.cols());
  all << xi, x;
  double out = dens(all) - dens(x);
  if (!log) out = std::exp(out);

  return out;
}

double DeterminantalPP::papangelou(const Point &xi, const std::list<Point> &x,
                                   bool log) {
  throw std::runtime_error(
      "DeterminantalPP::papangelou(const Point &xi, const std::list<Point> "
      "&x, bool log) NOT IMPLEMENTED");
  return stan::math::NEGATIVE_INFTY;
}

VectorXd DeterminantalPP::phi_star_rng() {
  VectorXd out(dim);
  for (int i = 0; i < dim; i++) {
    out(i) = uniform_rng(ranges(0, i), ranges(1, i), Rng::Instance().get());
  }
  return out;
}

double DeterminantalPP::phi_star_dens(VectorXd xi, bool log) {
  double out = phi_tildes.sum() / vol_range;
  if (log) out = std::log(out);

  return out;
}

void DeterminantalPP::update_hypers(const MatrixXd &active,
                                    const MatrixXd &non_active) {
  double rho_new = rho;

  if (rho_new != rho) eigen_decomposition();
};

double DeterminantalPP::log_det_Ctilde(const MatrixXd &x) {
  MatrixXd Ctilde(x.rows(), x.rows());

  // TODO: Ctilde is symmetric! Also the diagonal elements are identical!
  for (int l = 0; l < x.rows(); l++) {
    for (int m = 0; m < x.rows(); m++) {
      double aux = 0.0;
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        double dotprod = Kappas.row(kind).dot(x.row(l) - x.row(m));
        aux += phi_tildes[kind] * std::cos(2. * PI * dotprod);
      }
      Ctilde(l, m) = aux;
    }
  }
  return 2.0 * std::log(Ctilde.llt().matrixL().determinant());
}

// Sets Kappas, phis, phi_tildes and Ds
void DeterminantalPP::eigen_decomposition() {
  std::vector<double> k(2 * N + 1);
  for (int n = -N; n <= N; n++) {
    k[n + N] = n;
  }
  std::vector<std::vector<double>> kappas;
  if (dim == 1) {
    kappas.resize(k.size());
    for (int i = 0; i < k.size(); i++) kappas[i].push_back(k[i]);
  } else {
    kappas = cart_product(k, dim);
  }

  Kappas.resize(kappas.size(), dim);
  for (int i = 0; i < kappas.size(); i++) {
    Kappas.row(i) = Map<VectorXd>(kappas[i].data(), dim).transpose();
  }

  phis.resize(Kappas.rows());
  phi_tildes.resize(Kappas.rows());
  Ds = 0.0;

  double dim_ = 1.0 * dim;
  double log_alpha_max = 1.0 / dim_ *
                         (stan::math::lgamma(dim_ / nu + 1) - std::log(rho) -
                          stan::math::lgamma(dim_ / 2 + 1) - 0.5 * LOG_SQRT_PI);
  double alpha_max = std::exp(alpha_max);

  for (int i = 0; i < Kappas.rows(); i++) {
    phis[i] =
        std::pow(s, dim) * std::exp(-(s * alpha_max * Kappas.row(i).norm()));

    phi_tildes[i] = phis[i] / (1 - phis[i]);
    Ds += std::log(1 + phi_tildes[i]);
  }
}

double DeterminantalPP::rejection_sampling_M(int npoints) {
  throw std::runtime_error(
      "DeterminantalPP::rejection_sampling_M(int npoints) NOT IMPLEMENTED");
  return stan::math::NEGATIVE_INFTY;
}
