#include "base_pp.hpp"

BasePP::BasePP(const MatrixXd &ranges) : ranges(ranges) {
  dim = ranges.cols();
  diff_range = (ranges.row(1) - ranges.row(0)).transpose();
  vol_range = diff_range.prod();
}

void BasePP::set_ranges(const MatrixXd &ranges) {
  this->ranges = ranges;
  dim = ranges.cols();
  // std::cout << "ranges: \n" << ranges << std::endl;
  diff_range = (ranges.row(1) - ranges.row(0)).transpose();
  vol_range = diff_range.prod();

  initialize();
}

MatrixXd BasePP::sample_uniform(int npoints) {
  MatrixXd out(npoints, dim);
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < npoints; i++) {
      out(i, j) =
          uniform_rng(ranges(0, j), ranges(1, j), Rng::Instance().get());
    }
  }

  return out;
}

void BasePP::sample_given_active(const MatrixXd &active, MatrixXd *non_active,
                                 double psi_u) {
  int npoints = non_active->rows();
  double c_star_na = c_star * psi_u;
  birth_prob = std::log(c_star_na) - std::log(c_star_na + npoints);

  double rsecond = uniform_rng(0, 1, Rng::Instance().get());
  birth_arate = -1;
  if (std::log(rsecond) < birth_prob) {
    // BIRTH MOVE
    VectorXd xi = phi_star_rng();
    MatrixXd aux(active.rows() + npoints, dim);
    aux << active, *non_active;
    double pap = papangelou(xi, aux);
    birth_arate = pap - phi_star_dens(xi);

    double rthird = uniform_rng(0, 1, Rng::Instance().get());
    if (std::log(rthird) < birth_arate) {
      non_active->conservativeResize(npoints + 1, dim);
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

MatrixXd BasePP::sample_n_points(int npoints) {
  int max_steps = 1e6;
  MatrixXd out(npoints, dim);
  double logM = std::log(rejection_sampling_M(npoints));

  for (int i = 0; i < max_steps; i++) {
    double dens_q = 0;
    for (int k = 0; k < npoints; k++) {
      out.row(k) = phi_star_rng().transpose();
      dens_q += phi_star_dens(out.row(k).transpose(), true);
    }

    double arate = dens(out) - (logM + dens_q);
    double u = stan::math::uniform_rng(0.0, 1.0, Rng::Instance().get());
    if (std::log(u) < arate) return out;
  }

  std::cout << "MAXIMUM NUMBER OF ITERATIONS REACHED IN "
            << "BasePP::sample_n_points, returning the last value" << std::endl;

  return out;
}
