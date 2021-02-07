#ifndef ALLOC_MEANS_SAMPLER_IMP
#define ALLOC_MEANS_SAMPLER_IMP

namespace MCMCsampler {

// returns the ln of full-cond of Lambda|rest in current Lambda (lamb is vectorized)
template <typename T>
T MeansSamplerMala::means_target_function::operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & mean) const {

  using std::pow; using std::exp; using std::log;
  
  MatrixXd others(allmeans.rows() - 1, m_mcmc->dim_fact);
  others = delete_row(allmeans, ind_selected_mean);

  T output{0.};
  
  int n = m_mcmc->etas_by_clus[ind_selected_mean].size();

  MatrixXd cluster_etas(m_mcmc->dim_fact, n );
  for (int i = 0; i < n ; i++)
    cluster_etas.col(i) = m_mcmc->etas_by_clus[ind_selected_mean](i);
  
  output -= 0.5 * sum(multiply(m_mcmc->a_deltas[ind_selected_mean].get_cho_factor_eval(), (cluster_etas.colwise()-mean)).colwise().squaredNorm());

  
  MatrixXd others_trans(allmeans.rows() - 1 ,m_mcmc.dim_fact);
  for (int i = 0; i < others_trans.rows() ; i++){
        others_trans.row(i) = (m_mcmc.pp_mix->get_A() * others.row(i).transpose() + m_mcmc.pp_mix->get_b()).transpose();
  }
  Eigen::Matrix<T, 1, Eigen::Dynamic> mean_trans;
  mean_trans = (multiply(m_mcmc.pp_mix->get_A(), mean) + m_mcmc.pp_mix->get_b()).transpose();

     //std::cout<<"checkpoint 9"<<std::endl;

  Matrix<T,Dynamic,Dynamic> Ctilde(allmeans.rows(), allmeans.rows());
  const MatrixXd& Kappas = m_mcmc.pp_mix->get_kappas();

  // TODO: Ctilde is symmetric! Also the diagonal elements are identical!
  for (int l = 0; l < others.rows(); l++) {
    for (int m = 0; m < others.rows(); m++) {
      double aux = 0.0;
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        double dotprod = Kappas.row(kind).dot(others_trans.row(l) - others_trans.row(m));
        aux += m_mcmc->pp_mix->get_phi_tildes()(kind) * std::cos(2. * PI * dotprod);
      }
      Ctilde(l, m) = aux;
    }
  }

  for (int m = 0; m < others.rows(); m++) {
    T aux = 0.0;
    for (int kind = 0; kind < Kappas.rows(); kind++) {
        T dotprod = dot_preduct(Kappas.row(kind), mean_trans - others_trans.row(m));
        aux += m_mcmc->pp_mix->get_phi_tildes()(kind) * cos(2. * stan::math::pi() * dotprod);
    }
    Ctilde(allmeans.rows()-1, m) = aux;
    Ctilde(m, allmeans.rows()-1) = aux;
  }
 
  output += log_determinant(Ctilde);

  return output;
  
}

};


#endif