#include "simulate_straus.hpp"

Eigen::MatrixXd simulate_strauss_moller(const Eigen::MatrixXd &ranges,
                                        double beta, double gamma, double R) {
  gsl_rng *r;
  r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand() % 100000);

  // Initialise point pattern
  Point2Pattern ExamplePattern((ranges(0, 0) - 1) * 2, (ranges(1, 0) + 1) * 2,
                               (ranges(0, 1) - 1) * 2, (ranges(1, 1) + 1) * 2,
                               9, 9);

  Sampler ExampleSimulator;

  StraussProcess ExampleProcess(ranges(0, 0), ranges(1, 0),  // range of x
                                ranges(0, 1), ranges(1, 1),  // range of y
                                beta, gamma, R);

  // Generate perfect sample of Strauss process
  ExampleSimulator.Sim(&ExamplePattern, &ExampleProcess, r);

  return ExamplePattern.to_eigen_mat();
}

Eigen::MatrixXd simulate_strauss_our(StraussPP *pp) {
  // simulate a Poisson process on (ranges) with phi^* intensity
  int npoints = stan::math::poisson_rng(pp->get_cstar(), Rng::Instance().get());

  MatrixXd upper(npoints, pp->get_dim());
  if (npoints == 0) return upper;

  for (int i = 0; i < npoints; i++)
    upper.row(i) = pp->phi_star_rng().transpose();

  int nsteps = 2;
  bool has_coalesced = false;

  MatrixXd out;

  while (!has_coalesced) {
    std::cout << "nsteps: " << nsteps << std::endl; 
    std::tuple<MatrixXd, std::vector<VectorXd>, std::vector<VectorXd>, VectorXd,
               VectorXi, VectorXi>
        birth_death_out = run_backwards(upper, nsteps, pp);

    upper = std::get<0>(birth_death_out);
    MatrixXd lower = MatrixXd(0, pp->get_dim());

    for (int i = nsteps - 1; i >= 0; i--) {
      
      if (std::get<4>(birth_death_out)(i) == 1) {
        double rthird = std::get<3>(birth_death_out)(i);
        // std::cout << "rthird: " << rthird << std::endl;
        VectorXd xi_m = std::get<2>(birth_death_out)[i];
        if (std::log(rthird) <
            pp->papangelou(xi_m, lower) - pp->phi_star_dens(xi_m)) {
          //   std::cout << "adding point to upper" << std::endl;
          upper.conservativeResize(upper.rows() + 1, upper.cols());
          upper.row(upper.rows() - 1) = xi_m.transpose();
        }

        if (std::log(rthird) <
            pp->papangelou(xi_m, upper) - pp->phi_star_dens(xi_m)) {
          //   std::cout << "adding point to lower" << std::endl;
          lower.conservativeResize(lower.rows() + 1, lower.cols());
          lower.row(lower.rows() - 1) = xi_m.transpose();
        }
      } else {
        VectorXd eta_m = std::get<1>(birth_death_out)[i];
        if (upper.rows() > 0) {
          VectorXd diffs =
              (upper.rowwise() - eta_m.transpose()).rowwise().norm();
          int pos;
          diffs.minCoeff(&pos);
          if (diffs(pos) < 1e-3) {
            delete_row(&upper, pos);
          }
        }
        if (lower.rows() > 0) {
          VectorXd diffs =
              (lower.rowwise() - eta_m.transpose()).rowwise().norm();
          int pos;
          diffs.minCoeff(&pos);
          if (diffs(pos) < 1e-3) {
            delete_row(&lower, pos);
          }
        }
      }
    }

    if (upper.rows() == lower.rows()) {
      if ((upper - lower).norm() < 1e-5) {
        has_coalesced = true;
        out = upper;
      }
    }
    nsteps = nsteps * 2;
  }
  return out;
}

std::tuple<MatrixXd, std::vector<VectorXd>, std::vector<VectorXd>, VectorXd,
           VectorXi, VectorXi>
run_backwards(const MatrixXd &init, int nsteps, StraussPP *pp) {
  MatrixXd curr = init;
  VectorXd rthirds(nsteps);
  VectorXi birth_death(nsteps);  // 0 = birth, 1 = death, 2 = neither
  VectorXi death_ind(nsteps);

  std::vector<VectorXd> back_births(nsteps);
  std::vector<VectorXd> back_deaths(nsteps);

  std::vector<MatrixXd> states(nsteps + 1);
  states[0] = init;
  double cstar = pp->get_cstar();

  for (int m = 0; m < nsteps; m++) {
    double rsec = stan::math::uniform_rng(0, 1, Rng::Instance().get());
    double rthird = -1;
    int birth = 2;
    int removed = -1;
    if (rsec > cstar / (cstar + curr.rows())) {
      rthird = stan::math::uniform_rng(0, 1, Rng::Instance().get());
      if (curr.rows() > 0) {
        VectorXd probas = VectorXd::Ones(curr.rows()) / curr.rows();
        removed = categorical_rng(probas, Rng::Instance().get()) - 1;
        back_deaths[m] = curr.row(removed).transpose();
        delete_row(&curr, removed);
        birth = 1;
      }
    } else {
      VectorXd newpoint = pp->phi_star_rng();
      curr.conservativeResize(curr.rows() + 1, curr.cols());
      curr.row(curr.rows() - 1) = newpoint.transpose();
      birth = 0;
      back_births[m] = newpoint;
    }
    rthirds(m) = rthird;
    birth_death(m) = birth;
    // states[m + 1] = curr;
    death_ind(m) = removed;
  }
  return std::make_tuple(curr, back_births, back_deaths, rthirds, birth_death,
                         death_ind);
}