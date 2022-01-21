#ifndef LAMBDA_SAMPLER_IMP_HPP
#define LAMBDA_SAMPLER_IMP_HPP

namespace MCMCsampler {

// returns the ln of full-cond of Lambda|rest in current Lambda (lamb is vectorized)
template <typename T>
T LambdaSamplerMala::lambda_target_function::operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & lamb) const {

  using std::pow; using std::exp; using std::log;
  //std::cout<<"checkpoint 1"<<std::endl;
  //Matrix<T,Dynamic,Dynamic> lamb_mat(dim_data,dim_fact);
  Matrix<T,Dynamic,Dynamic> lamb_mat=Map<const Matrix<T,Dynamic,Dynamic>>(lamb.data(),m_mcmc.get_dim_data(),m_mcmc.get_dim_fact());

  /*
  for (int j=0 ;j<dim_fact;j++){
  lamb_mat.col(j) = lamb.segment(j*dim_data,j*dim_data+dim_data);
  }
*/
 T output{0.};
 // std::cout<<"checkpoint 2"<<std::endl;

 // ######### FIRST TERM

  // TO BE SUBSTITUTED WITH CALL TO TEMPLATE FUNCTION WORKING FOR DOUBLE AND T TYPES
  /*
  Matrix<T,Dynamic,Dynamic> f0(m_mcmc->etas.transpose());
    std::cout<<"checkpoint 2a"<<std::endl;
  Matrix<T,Dynamic,Dynamic> f1(lamb_mat*f0);
      std::cout<<"checkpoint 2b"<<std::endl;
  Matrix<T,Dynamic,Dynamic> f2(m_mcmc->data.transpose() - f1);
      std::cout<<"checkpoint 2c"<<std::endl;
  Matrix<T,Dynamic,1> f3(m_mcmc->sigma_bar.array().sqrt().matrix());
      std::cout<<"checkpoint 2d"<<std::endl;
  Matrix<T,Dynamic,Dynamic> f4(f3.asDiagonal());
      std::cout<<"checkpoint 2e"<<std::endl;
  Matrix<T,Dynamic,Dynamic> f5(f4*f2);
      std::cout<<"checkpoint 2f"<<std::endl;
  Matrix<T,Dynamic,Dynamic> f6(f5.colwise().squaredNorm());
      std::cout<<"checkpoint 2g"<<std::endl;
  T f7(f6.sum());
  std::cout<<"checkpoint 3"<<std::endl;
*/

  output += -0.5 * sum(multiply(m_mcmc.get_sigma_bar().array().sqrt().matrix().asDiagonal(),subtract(m_mcmc.get_data().transpose(),multiply(lamb_mat, m_mcmc.get_etas().transpose()))).colwise().squaredNorm());

  //output += -0.5 * (m_mcmc->sigma_bar.array().sqrt().matrix().asDiagonal()*(m_mcmc->data.transpose() - lamb_mat*m_mcmc->etas.transpose())).colwise().squaredNorm().sum();
  //output += multiply(lambda_mat,etas.transpose());

  // ############# SECOND TERM
  //std::cout<<"checkpoint 4"<<std::endl;

  T esp_fact = -2*pow(stan::math::pi(),2)*pow(exp(log_determinant_spd(crossprod(lamb_mat))),1.0/m_mcmc.get_dim_fact())*pow(m_mcmc.pp_mix->get_c(),-2.0/m_mcmc.get_dim_fact());

  T Ds(0.);
  Matrix<T,Dynamic,1> phis;
  Matrix<T,Dynamic,1> phi_tildes;
  const MatrixXd& Kappas = m_mcmc.pp_mix->get_kappas();
  phis.resize(Kappas.rows());
  phi_tildes.resize(Kappas.rows());
  double s = m_mcmc.pp_mix->get_s();
  for (int i = 0; i < Kappas.rows(); i++) {
     // std::cout<<"checkpoint 7"<<std::endl;

    phis(i) = s*exp(esp_fact * dot_product(Kappas.row(i),mdivide_left_spd(crossprod(lamb_mat), Kappas.row(i).transpose() )) );

    phi_tildes(i) = phis(i) / (1 - phis(i));

  }
  Ds = sum(log(1 + phi_tildes.array()));

  int n_means=m_mcmc.get_num_a_means()+m_mcmc.get_num_na_means();
  MatrixXd mu_trans(n_means,m_mcmc.get_dim_fact());
  for (int i = 0; i < n_means; i++){
        mu_trans.row(i) = (m_mcmc.pp_mix->get_A() * m_mcmc.get_all_means().row(i).transpose() + m_mcmc.pp_mix->get_b()).transpose();
  }
     //std::cout<<"checkpoint 9"<<std::endl;

  Matrix<T,Dynamic,Dynamic> Ctilde(mu_trans.rows(), mu_trans.rows());

  for (int l = 0; l < mu_trans.rows(); l++) {
    for (int m = l; m < mu_trans.rows(); m++) {
      T aux = 0.0;
      RowVectorXd vec = mu_trans.row(l) - mu_trans.row(m);
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        double dotprod = Kappas.row(kind).dot(vec);
        aux += phi_tildes(kind) * cos(2. * stan::math::pi() * dotprod);
      }
      Ctilde(l, m) = aux;
      if (l!=m) Ctilde(m,l) = aux;

    }
  }

  output += -Ds - log(1-exp(-Ds)) + log_determinant(Ctilde);

  //######## THIRD TERM
  /*
  Matrix<T,Dynamic,Dynamic> f9((m_mcmc->Psi.array()) * (m_mcmc->Phi.array().square()));
  Matrix<T,Dynamic,Dynamic> f10(lamb_mat.array().square()/f9.array());
  T f11 = f10.sum();
  T f12 =  f11 * (- 0.5 / (m_mcmc->tau*m_mcmc->tau));*/

  output += (- 0.5 / (m_mcmc.get_tau()*m_mcmc.get_tau())) * sum(elt_divide(lamb_mat.array().square(),(m_mcmc.get_Psi().array()) * (m_mcmc.get_Phi().array().square())));

  return output;
}



}

#endif
