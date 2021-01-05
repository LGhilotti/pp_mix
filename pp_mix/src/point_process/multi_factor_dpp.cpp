#include "multi_factor_dpp.hpp"


MultiDpp::MultiDpp(const MatrixXd &ranges, int N, double c, double s): BaseDeterminantalPP(ranges,N,c,s) {

  std::cout<<"MultiDpp constructor"<<std::endl;
  compute_Kappas();
  phis.resize(Kappas.rows());
  phi_tildes.resize(Kappas.rows());
  phis_tmp.resize(Kappas.rows());
  phi_tildes_tmp.resize(Kappas.rows());
  std::cout<<"end Multi constructor"<<std::endl;

}


void MultiDpp::set_decomposition(const MatrixXd * lambda) {

  Lambda = lambda;
  compute_eigen_and_cstar(&Ds, &phis, &phi_tildes, &c_star, lambda);
  std::cout<<"decomposition set"<<std::endl;
  return;

}



void MultiDpp::compute_eigen_and_cstar(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda){

  std::cout << "compute eigen and cstar! "<<std::endl;

  *D_ = 0.0;
  *C_star_ = 0.0;

  LLT<MatrixXd> M ((*lambda).transpose() * (*lambda));
  // compute determinant of Lambda^T Lambda
  double det = std::pow(M.matrixL().determinant(),2);

  double esp_fact = -2*std::pow(stan::math::pi(),2)*std::pow(det,1.0/dim)*std::pow(c,-2.0/dim);
  for (int i = 0; i < Kappas.rows(); i++) {
    VectorXd sol = M.solve(Kappas.row(i).transpose());
    double dot_prod = (Kappas.row(i)).dot(sol);
    (*Phis_)(i) = s*std::exp(esp_fact*dot_prod);

    (*Phi_tildes_)(i) = (*Phis_)(i) / (1 - (*Phis_)(i));
    *D_ += std::log(1 + (*Phi_tildes_)(i));
    *C_star_ += (*Phi_tildes_)(i);
  }

  return;

}



void MultiDpp::decompose_proposal(const MatrixXd& lambda) {

  compute_eigen_and_cstar(&Ds_tmp, &phis_tmp, &phi_tildes_tmp, &c_star_tmp, &lambda);
  return;

}


void MultiDpp::update_decomposition_from_proposal() {

  std::swap(Ds, Ds_tmp);
  phis.swap(phis_tmp);
  phi_tildes.swap(phi_tildes_tmp);
  std::swap(c_star, c_star_tmp);
  return;
}


// compute just once the grid for summation over Z^dim
void MultiDpp::compute_Kappas() {

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

  std::cout<<"Kappas: "<<Kappas<<std::endl;

  return;

}


double MultiDpp::dens_cond_in_proposal(const MatrixXd& x, bool log) {

  double out = ln_dens_process(x, Ds_tmp, phis_tmp, phi_tildes_tmp, c_star_tmp);
  out -= std::log(1-std::exp(-Ds_tmp));

  if (!log) out=std::exp(out);

  return out;

}
