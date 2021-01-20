#ifndef MALA_CONDITIONAL_MCMC_IMP_HPP
#define MALA_CONDITIONAL_MCMC_IMP_HPP

namespace Mala {

// returns the ln of full-cond of Lambda|rest in current Lambda (lamb is vectorized)
template <typename T>
T MalaMultiMCMC::target_function::operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & lamb) const {

  using std::pow; using std::exp; using std::log;

  //Matrix<T,Dynamic,Dynamic> lamb_mat(dim_data,dim_fact);
  Matrix<T,Dynamic,Dynamic> lamb_mat=Map<const Matrix<T,Dynamic,Dynamic>>(lamb.data(),m_mcmc->dim_data,m_mcmc->dim_fact);

  /*
  for (int j=0 ;j<dim_fact;j++){
  lamb_mat.col(j) = lamb.segment(j*dim_data,j*dim_data+dim_data);
  }
*/
 T output{0.};
  // TO BE SUBSTITUTED WITH CALL TO TEMPLATE FUNCTION WORKING FOR DOUBLE AND T TYPES
  Matrix<T,Dynamic,Dynamic> f0(m_mcmc->etas.transpose());
  Matrix<T,Dynamic,Dynamic> f1(lamb_mat*f0);
  Matrix<T,Dynamic,Dynamic> f2(m_mcmc->data.transpose() - f1);
  Matrix<T,Dynamic,1> f3(m_mcmc->sigma_bar.array().sqrt().matrix());
  Matrix<T,Dynamic,Dynamic> f4(f3.asDiagonal());
  Matrix<T,Dynamic,Dynamic> f5(f4*f2);
  Matrix<T,Dynamic,Dynamic> f6(f5.colwise().squaredNorm());
  T f7(f6.sum());

  output += -0.5 * f7;
  //output += -0.5 * (m_mcmc->sigma_bar.array().sqrt().matrix().asDiagonal()*(m_mcmc->data.transpose() - lamb_mat*m_mcmc->etas.transpose())).colwise().squaredNorm().sum();
  //output += multiply(lambda_mat,etas.transpose());
/*
  LLT<Matrix<T,Dynamic,Dynamic>> M ((lamb_mat).transpose() * (lamb_mat));
  T det = exp(log_determinant_ldlt(M));
  double c = 10. ;
  double s = 0.8 ;
  T esp_fact = -2*pow(stan::math::pi(),2)*pow(det,1.0/dim_fact)*pow(c,-2.0/dim_fact);
  MatrixXd Kappas = pp_mix->get_kappas();

  T Ds(0.);
  Matrix<T,Dynamic,1> phis;
  MatrixXd<T,Dynamic,1> phi_tildes;
  phis.resize(Kappas.rows());
  phi_tildes.resize(Kappas.rows());

  for (int i = 0; i < Kappas.rows(); i++) {
    Matrix<T,Dynamic,1> sol = M.solve(Kappas.row(i).transpose());
    T dot_prod = (Kappas.row(i)).dot(sol);
    phis(i) = s*exp(esp_fact*dot_prod);

    phi_tildes(i) = phis(i) / (1 - phis(i));
    Ds += log(1 + phi_tildes(i));
  }*/
  // I MUST ADD THE LOG DET CTILDE (and also last term)
/*
  for (int l = 0; l < .rows(); l++) {
    for (int m = 0; m < x.rows(); m++) {
      double aux = 0.0;
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        double dotprod = Kappas.row(kind).dot(x.row(l) - x.row(m));
        aux += phi_tildes_p[kind] * std::cos(2. * PI * dotprod);
      }
      Ctilde(l, m) = aux;
    }
  }
  */
//  output += -Ds - log(1-exp(-Ds));

  return output;
}



};
#endif
