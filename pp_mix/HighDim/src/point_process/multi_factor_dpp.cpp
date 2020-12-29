#include "multi_factor_dpp.hpp"

// Sets the pointer to Lambda and sets phis, phi_tildes, Ds and c_star
// Once set these parameters, no more difference for multi/uni factor
void MultiDpp::compute_eigen_and_cstar(const MatrixXd * Lamb) override {
  std::cout << "compute eigen and cstar! "<<std::endl;

  Lambda = Lamb;

  Ds = 0.0;
  c_star = 0.0;

  LLT<MatrixXd> M ((*Lambda).transpose() * (*Lambda));
  // compute determinant of Lambda^T Lambda
  double det = std::pow(M.matrixL().determinant(),2);

  double esp_fact = -2*std::pow(PI,2)*std::pow(det,1.0/dim)*std::pow(c,-2.0/dim);
  for (int i = 0; i < Kappas.rows(); i++) {
    VectorXd sol = M.solve(Kappas.row(i).transpose());
    double dot_prod = (Kappas.row(i)).dot(sol);
    phis[i] = s*std::exp(esp_fact*dot_prod);

    phi_tildes[i] = phis[i] / (1 - phis[i]);
    Ds += std::log(1 + phi_tildes[i]);
    c_star += phi_tildes[i];
  }

  std::cout << "Ds: "<<this->Ds<<std::endl;
  std::cout << "cstar: "<<this->c_star<<std::endl;

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
