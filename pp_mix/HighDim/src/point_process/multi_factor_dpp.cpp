#include "multi_factor_dpp.hpp"



void MultiDpp::set_decomposition(const MatrixXd * lambda) override {

  Lambda = lambda;
  compute_eigen_and_cstar(&Ds, &phis, &phi_tildes, &c_star, lambda);
  return;

}



void MultiDpp::compute_eigen_and_cstar(double * D, VectorXd * Phis, VectorXd * Phi_tildes, double * C_star, const MatrixXd * lambda){

  std::cout << "compute initial eigen and cstar! "<<std::endl;

  D -> 0.0;
  C_star -> 0.0;

  LLT<MatrixXd> M ((*lambda).transpose() * (*lambda));
  // compute determinant of Lambda^T Lambda
  double det = std::pow(M.matrixL().determinant(),2);

  double esp_fact = -2*std::pow(PI,2)*std::pow(det,1.0/dim)*std::pow(c,-2.0/dim);
  for (int i = 0; i < Kappas.rows(); i++) {
    VectorXd sol = M.solve(Kappas.row(i).transpose());
    double dot_prod = (Kappas.row(i)).dot(sol);
    (*Phis)[i] = s*std::exp(esp_fact*dot_prod);

    (*Phi_tildes)[i] = (*Phis)[i] / (1 - (*Phis)[i]);
    *D += std::log(1 + (*Phi_tildes)[i]);
    *C_star += (*Phi_tildes)[i];
  }

  return;

}



void MultiDpp::decompose_proposal(const MatrixXd& lambda) override {

  compute_eigen_and_cstar(&Ds_tmp, &phis_tmp, &phi_tildes_tmp, &c_star_tmp, &lambda);
  return;

}



// compute just once the grid for summation over Z^dim
void MultiDpp::compute_Kappas() override {

  std::cout << "compute Kappas!" <<std::endl;

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
  return;

}


double MultiDpp::dens_cond_in_proposal(const MatrixXd& x, bool log) override {

  double out = ln_dens_process(x, Ds_tmp, phis_tmp, phi_tildes_tmp, c_star_tmp);
  out -= std::log(1-std::exp(-Ds_tmp));

  if (!log) out=std::exp(out);

  return out;

}
