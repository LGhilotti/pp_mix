#include "base_determinantalPP.hpp"

#include <numeric>

using stan::math::LOG_SQRT_PI;
double PI = stan::math::pi();


BaseDeterminantalPP::BaseDeterminantalPP(const MatrixXd &ranges, int N, double c, double s):
  ranges(ranges), N(N), c(c), s(s) {

  std::cout << "constructor!"<<std::endl;
  dim = ranges.cols();
  diff_range = (ranges.row(1) - ranges.row(0)).transpose();
  vol_range = diff_range.prod();

  A = MatrixXd::Zero(dim, dim);
  b = VectorXd::Zero(dim);
  for (int i = 0; i < dim; i++) {
    A(i, i) = 1.0 / (ranges(1, i) - ranges(0, i));
    b(i) = -A(i, i) * (ranges(1, i) + ranges(0, i)) / 2.0;
  }
  // std::cout << "ranges: "<<this->ranges<<std::endl;
  // std::cout << "N: "<<this->N<<std::endl;
  // std::cout << "c: "<<this->c<<std::endl;
  // std::cout << "s: "<<this->s<<std::endl;
  // std::cout << "dim: "<<this->dim<<std::endl;
  // std::cout << "diff_range: "<<this->diff_range<<std::endl;
  // std::cout << "vol_range: "<<this->vol_range<<std::endl;

  return;
}


double BaseDeterminantalPP::dens_cond(const MatrixXd& x, bool log) {

  double out = ln_dens_process(x, Ds, phis, phi_tildes, c_star);
  out -= std::log(1-std::exp(-Ds));

  if (!log) out = std::exp(out);

  return out;

}

double BaseDeterminantalPP::dens(const MatrixXd &x, bool log) {

  double out = ln_dens_process(x, Ds, phis, phi_tildes, c_star);

  if (!log) out = std::exp(out);

  return out;

}


double BaseDeterminantalPP::ln_dens_process(const MatrixXd& x, double Ds_p, const VectorXd& phis_p,
            const VectorXd& phi_tildes_p, double c_star_p){

  double out;
  int n;

  // check if it's jut one point
  if ((x.size() == 1 && dim == 1) || (x.rows() == 1 & dim > 1) ||
      (x.cols() == 1 && dim > 1)) {
    n = 1;
    out =
        -1.0 * n * std::log(vol_range) - Ds_p + std::log(c_star_p);
  }
  else {
    int n = x.rows();
    bool check_range = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        if (x(i, j) < ranges(0, j) || x(i, j) > ranges(1, j))
          check_range = false;
      }
    }
    if (check_range) {
      out = -1.0 * n * std::log(vol_range) - Ds_p;

      // Transform data points to be in the unit cube centered in 0
      MatrixXd xtrans(n,x.cols());

      for (int i = 0; i < n; i++)
        xtrans.row(i) = (A * x.row(i).transpose() + b).transpose();

      // std::cout << "xtrans " << xtrans.transpose() << std::endl;
      out += log_det_Ctilde(xtrans, phi_tildes_p);
    } else {
      out = stan::math::NEGATIVE_INFTY;
    }
  }
  // std::cout << "dens: " << out << std::endl;

  return out;

}


double BaseDeterminantalPP::log_det_Ctilde(const MatrixXd &x, const VectorXd& phi_tildes_p) {
  MatrixXd Ctilde(x.rows(), x.rows());

  // TODO: Ctilde is symmetric! Also the diagonal elements are identical!
  for (int l = 0; l < x.rows(); l++) {
    for (int m = 0; m < x.rows(); m++) {
      double aux = 0.0;
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        double dotprod = Kappas.row(kind).dot(x.row(l) - x.row(m));
        aux += phi_tildes_p[kind] * std::cos(2. * PI * dotprod);
      }
      Ctilde(l, m) = aux;
    }
  }
  return 2.0 * std::log(Ctilde.llt().matrixL().determinant());
}


double BaseDeterminantalPP::papangelou(const VectorXd& xi, const MatrixXd &x, bool log) {
  int n = 1 + x.rows();
  MatrixXd all( n , x.cols());
  all << x, xi.transpose();

  // Transform data points to be in the unit cube centered in 0
  MatrixXd alltrans(n,all.cols());
  for (int i = 0; i < n; i++)
    alltrans.row(i) = (A * all.row(i).transpose() + b).transpose();

  double out = -1.0*std::log(vol_range)+ log_det_Ctilde(alltrans, phi_tildes) - log_det_Ctilde(alltrans.topRows(n-1), phi_tildes);

  if (!log) out = std::exp(out);

  return out;
}



MatrixXd BaseDeterminantalPP::sample_uniform(int npoints) {
  MatrixXd out(npoints, dim);
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < npoints; i++) {
      out(i, j) =
          uniform_rng(ranges(0, j), ranges(1, j), Rng::Instance().get());
    }
  }

  return out;
}


VectorXd BaseDeterminantalPP::phi_star_rng() {
  VectorXd out(dim);
  for (int i = 0; i < dim; i++) {
    out(i) = uniform_rng(ranges(0, i), ranges(1, i), Rng::Instance().get());
  }
  return out;
}

double BaseDeterminantalPP::phi_star_dens(VectorXd xi, bool log) {
  double out = std::log(c_star) - std::log(vol_range);
  if (!log) out = std::exp(out);

  return out;
}


void BaseDeterminantalPP::sample_nonalloc_fullcond(MatrixXd *non_active, const MatrixXd &active,
                                 double psi_u) {
  std::cout<<"inside sample_given_active"<<std::endl;
  int npoints = non_active->rows();
  std::cout<<"rows: " <<non_active->rows()<<std::endl;
  std::cout<<"columns: " <<non_active->cols()<<std::endl;

  double c_star_na = c_star * psi_u;
  birth_prob = std::log(c_star_na) - std::log(c_star_na + npoints);

  double rsecond = uniform_rng(0, 1, Rng::Instance().get());
  birth_arate = -1;
  if (std::log(rsecond) < birth_prob) {
    // BIRTH MOVE
    VectorXd xi = phi_star_rng();

    // compute prob of acceptance of the new birth
    MatrixXd aux(active.rows() + npoints, dim);
    std::cout<<"defined aux"<<std::endl;
    aux << active, *non_active;
    std::cout<<"filled aux: "<<aux<<std::endl;
    std::cout<<"rows= "<<aux.rows()<<std::endl;
    std::cout<<"cols= "<<aux.cols()<<std::endl;

    double pap = papangelou(xi, aux);
    std::cout<<"done papan"<<std::endl;
    birth_arate = pap - phi_star_dens(xi);

    double rthird = uniform_rng(0, 1, Rng::Instance().get());
    if (std::log(rthird) < birth_arate) {
      std::cout<<"entered"<<std::endl;
      non_active->conservativeResize(npoints + 1, dim);
      std::cout<<"done conservative"<<std::endl;
      non_active->row(npoints) = xi;
    }
  } else {
    // Death Move
    if (npoints == 0) return;

    VectorXd probas = VectorXd::Ones(npoints) / npoints;
    int ind = categorical_rng(probas, Rng::Instance().get()) - 1;

    delete_row(non_active, ind);
  }

  return;
}
