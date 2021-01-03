#include "uni_factor_dpp.hpp"


UniDpp::UniDpp(const MatrixXd &ranges, int N, double c, double s): BaseDeterminantalPP(ranges,N,c,s) {

  compute_Kappas();
  phis.resize(Kappas.rows());
  phi_tildes.resize(Kappas.rows());

}

void UniDpp::set_decomposition(const MatrixXd * lambda) {

  compute_eigen_and_cstar();
  return;

}


void UniDpp::compute_eigen_and_cstar() {

  std::cout << "compute initial eigen and cstar! "<<std::endl;

  Ds = 0.0;
  c_star = 0.0;

  double esp_fact = -2*std::pow(stan::math::pi(),2)/std::pow(c,2.0/dim);
  phis = (esp_fact*Kappas.array().square()).exp() * s;
  phi_tildes = phis.array()/(1-phis.array()) ;
  Ds = std::log( (1 + phi_tildes.array()).prod() ) ;
  c_star = phi_tildes.sum() ;

  return;

}


// compute just once the grid for summation over Z
void UniDpp::compute_Kappas() {

  std::cout << "compute Kappas!" <<std::endl;

  Kappas.resize(2*N +1, 1);

  for (int n = -N; n <= N; n++) {
    Kappas(n + N) = n;
  }

  return;

}


double UniDpp::dens_cond_in_proposal(const MatrixXd& x, bool log) {

  return dens_cond(x,log);

}
