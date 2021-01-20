#include "tmp_conditional_mcmc.hpp"

#include "tmp_conditional_mcmc_imp.hpp"

namespace Test {
MultivariateConditionalMCMC::MultivariateConditionalMCMC(BaseDeterminantalPP *pp_mix,
                                                         BasePrec *g,
                                                         const Params &params,
                                                        // const MatrixXd& lambda,
                                                         //const VectorXd& SigBar,
                                                         //const MatrixXd& Etas,
                                                         double p_m_sigma, double p_l_sigma)
    : ConditionalMCMC<BaseMultiPrec, PrecMat, VectorXd>() {
  std::cout<<"begin multiMCMC constructor"<<std::endl;
  set_pp_mix(pp_mix);
  set_prec(dynamic_cast<BaseMultiPrec *>(g));
  set_params(params);
  //Lambda = lambda;
//  sigma_bar = SigBar;
//  etas = Etas;
  prop_means_sigma = p_m_sigma;
  prop_lambda_sigma = p_l_sigma;


}

void MultivariateConditionalMCMC::initialize_etas(const MatrixXd &dat) {

  LLT<MatrixXd> M (Lambda.transpose() * Lambda);

  etas = (M.solve((dat*Lambda).transpose())).transpose();

  return;

}


void MultivariateConditionalMCMC::initialize_allocated_means() {
  int init_n_clus = 10;
  std::vector<VectorXd> in = proj_inside();

  if (init_n_clus >= in.size()) {
    a_means.resize(in.size(), dim_fact);
    for (int i=0; i < in.size(); i++)
      a_means.row(i) = in[i].transpose();
  } else {
    a_means.resize(init_n_clus, dim_fact);
    std::vector<int> index(in.size());
    std::iota(index.begin(), index.end(), 0);
    std::random_shuffle(index.begin(), index.begin() + in.size());
    for (int i=0; i < init_n_clus; i++) {
      a_means.row(i) = in[index[i]].transpose();
    }
  }
  return;
}


void MultivariateConditionalMCMC::sample_etas() {

    MatrixXd M0(Lambda.transpose() * sigma_bar.asDiagonal());
    MatrixXd M1( M0 * Lambda);
    std::vector<MatrixXd> Sn_bar(a_means.rows());
    // type LLT for solving systems of equations
    std::vector<LLT<MatrixXd>> Sn_bar_cho (a_means.rows());

    for (int i=0; i < a_means.rows(); i++){
      Sn_bar[i]=M1+a_deltas[i].get_prec();
      Sn_bar_cho[i]= LLT<MatrixXd>(Sn_bar[i]);
    }

    MatrixXd M2(M0*data.transpose());
    MatrixXd G(ndata,dim_fact);
    // known terms of systems is depending on the single data
    for (int i=0; i < a_means.rows(); i++){
      MatrixXd B(dim_fact,obs_by_clus[i].size());
      B = (a_deltas[i].get_prec() * a_means.row(i).transpose()).replicate(1,B.cols());
      B +=M2(all,obs_by_clus[i]);
      // each modified row has solution for points in the cluster.
      G(obs_by_clus[i],all)=(Sn_bar_cho[i].solve(B)).transpose();
    }

    // Here, G contains (in each row) mean of full-cond, while precisions have to be taken from Sn_bar
    // Now, I sample each eta from the full-cond multi-normal
    for (int i=0; i < ndata; i++){
      etas.row(i)=multi_normal_prec_rng(G.row(i).transpose(), Sn_bar[clus_alloc(i)], Rng::Instance().get());
    }
    return;
}


void MultivariateConditionalMCMC::sample_Lambda() {
  //std::cout<<"sample Lambda"<<std::endl;
  // Current Lambda (here are the means) are expanded to vector<double> column major
  MatrixXd prop_lambda = Map<MatrixXd>(normal_rng( std::vector<double>(Lambda.data(), Lambda.data() + Lambda.size()) ,
              std::vector<double>(dim_data*dim_fact, prop_lambda_sigma), Rng::Instance().get()).data() , dim_data, dim_fact);
  // DEBUG
  //std::cout<<"Proposal Lambda: \n"<<prop_lambda<<std::endl;
//  std::cout<<"Proposed Lambda"<<std::endl;

  tot_sampled_Lambda += 1;
  // we use log for each term
  double curr_lik, prop_lik;
  curr_lik = -0.5 * compute_exp_lik(Lambda);
  prop_lik = -0.5 * compute_exp_lik(prop_lambda);
  // DEBUG
  //std::cout<<"curr_lik = "<<curr_lik<<"  ; prop_lik = "<<prop_lik<<std::endl;

  double curr_prior_cond_process, prop_prior_cond_process;
  MatrixXd means(a_means.rows()+na_means.rows(),dim_fact);
  means << a_means, na_means;

  pp_mix->decompose_proposal(prop_lambda);

  curr_prior_cond_process = pp_mix->dens_cond(means, true);
  prop_prior_cond_process = pp_mix->dens_cond_in_proposal(means, true);
  // DEBUG
  //std::cout<<"curr_p_c_p = "<<curr_prior_cond_process<<"  ; prop_p_c_p = "<<prop_prior_cond_process<<std::endl;

  double curr_prior_lambda, prop_prior_lambda;
  curr_prior_lambda = compute_exp_prior(Lambda);
  prop_prior_lambda = compute_exp_prior(prop_lambda);
  // DEBUG
 //std::cout<<"curr_prior_lambda = "<<curr_prior_lambda<<" ; prop_prior_lambda = "<<prop_prior_lambda<<std::endl;

  double curr_dens, prop_dens, log_ratio;
  curr_dens = curr_lik + curr_prior_cond_process + curr_prior_lambda;
  prop_dens = prop_lik + prop_prior_cond_process + prop_prior_lambda;
  // DEBUG
//  std::cout<<"curr_dens = "<<curr_dens<<" ; prop_dens = "<<prop_dens<<std::endl;

  log_ratio = prop_dens - curr_dens;
  // DEBUG
  //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

  if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < log_ratio){
    //ACCEPTED
    acc_sampled_Lambda += 1;
    Lambda.swap(prop_lambda);
    pp_mix->update_decomposition_from_proposal();
    //std::cout<<"accepted Lambda"<<std::endl;
  }

  return;
}

/*
VectorXd MultivariateConditionalMCMC::compute_grad_for_clus(
    int clus, const VectorXd &mean) {
  VectorXd grad = VectorXd::Zero(dim_data);
  for (const VectorXd datum : data_by_clus[clus]) grad += datum - mean;
  // TODO FIXME
  // grad += a_deltas[clus].get_prec() * (datum - mean);

  return grad;
}
*/
void MultivariateConditionalMCMC::get_state_as_proto(
    google::protobuf::Message *out_) {
  using namespace google::protobuf::internal;

  MultivariateMixtureState *out = down_cast<MultivariateMixtureState *>(out_);
  out->set_ma(a_means.rows());
  out->set_mna(na_means.rows());
  out->set_mtot(a_means.rows() + na_means.rows());

  for (int i = 0; i < a_means.rows(); i++) {
    EigenVector *mean;
    EigenMatrix *prec;
    mean = out->add_a_means();
    prec = out->add_a_deltas();

    to_proto(a_means.row(i).transpose(), mean);
    to_proto(a_deltas[i].get_prec(), prec);
  }

  for (int i = 0; i < na_means.rows(); i++) {
    EigenVector *mean;
    EigenMatrix *prec;
    mean = out->add_na_means();
    prec = out->add_na_deltas();

    to_proto(na_means.row(i).transpose(), mean);
    to_proto(na_deltas[i].get_prec(), prec);
  }

  to_proto(a_jumps, out->mutable_a_jumps());
  to_proto(na_jumps, out->mutable_na_jumps());

  *out->mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() + ndata};

  out->set_u(u);

  for (int i = 0; i < ndata; i++) {
    EigenVector * eta;
    eta = out->add_etas();

    to_proto(etas.row(i).transpose(), eta);
  }

  to_proto(sigma_bar, out->mutable_sigma_bar());

  LambdaBlock lb;
  lb.set_tau(tau);
  to_proto(Phi, lb.mutable_phi());
  to_proto(Psi, lb.mutable_psi());
  to_proto(Lambda, lb.mutable_lambda());
  out->mutable_lamb_block()->CopyFrom(lb);

  return;


}

void MultivariateConditionalMCMC::print_data_by_clus(int clus) {
  for (const int &d : obs_by_clus[clus])
    std::cout << data.row(d).transpose() << std::endl;
}

/////////////////////////////////
// UNIVARIATE CONDITIONAL MCMC //
/////////////////////////////////

UnivariateConditionalMCMC::UnivariateConditionalMCMC(BaseDeterminantalPP *pp_mix,
                                                     BasePrec *g,
                                                     const Params &params)
    : ConditionalMCMC<BaseUnivPrec, double, double>() {
  set_pp_mix(pp_mix);
  set_prec(dynamic_cast<BaseUnivPrec *>(g));
  set_params(params);
  /*
  min_proposal_sigma = 0.1;
  max_proposal_sigma = 1.0;
  */
}


void UnivariateConditionalMCMC::initialize_etas(const MatrixXd &dat) {

  etas = (dat*Lambda)/Lambda.squaredNorm();

  return;

}


void UnivariateConditionalMCMC::initialize_allocated_means() {
  int init_n_clus = 4;
  std::vector<VectorXd> in = proj_inside();

  if (init_n_clus >= in.size()) {
    a_means.resize(in.size(), 1);
    for (int i=0; i < in.size(); i++)
      a_means(i,0) = in[i](0);
  } else {
    a_means.resize(init_n_clus, 1);
    std::vector<int> index(in.size());
    std::iota(index.begin(), index.end(), 0);
    std::random_shuffle(index.begin(), index.begin() + in.size());
    for (int i=0; i < init_n_clus; i++) {
      a_means(i,0) = in[index[i]](0);
    }
  }
  return;
}

void UnivariateConditionalMCMC::sample_etas() {

    RowVectorXd M0(Lambda.transpose() * sigma_bar.asDiagonal());
    double M1( (M0 * Lambda)(0) );
    std::vector<double> Sn_bar(a_means.rows());

    for (int i=0; i < a_means.rows(); i++){
      Sn_bar[i]=M1+a_deltas[i];
    }

    RowVectorXd M2(M0*data.transpose());
    RowVectorXd G(ndata);
    // known terms of systems is depending on the single data
    for (int i=0; i < a_means.rows(); i++){
      RowVectorXd B(obs_by_clus[i].size());
      B = (a_deltas[i] * a_means(i,0)) * RowVectorXd::Ones(B.size());
      B +=M2(obs_by_clus[i]);
      // each modified element has solution for points in the cluster.
      G(obs_by_clus[i]) = B/Sn_bar[i];
    }

    // Here, G contains means of full-cond, while precisions have to be taken from Sn_bar
    // Now, I sample each eta from the full-cond normal
    for (int i=0; i < ndata; i++){
      etas(i,0)=normal_rng(G(i), 1.0/Sn_bar[clus_alloc(i)], Rng::Instance().get());
    }
    return;
}


void UnivariateConditionalMCMC::sample_Lambda() {

  MatrixXd prop_lambda = Map<MatrixXd>(normal_rng(std::vector<double>(Lambda.data(), Lambda.data() + Lambda.size()),
              std::vector<double>(dim_data, prop_lambda_sigma), Rng::Instance().get()).data(), dim_data,1);

  tot_sampled_Lambda += 1;
  // we use log for each term
  double curr_lik, prop_lik;
  curr_lik = -0.5 * compute_exp_lik(Lambda);
  prop_lik = -0.5 * compute_exp_lik(prop_lambda);

  double curr_prior_lambda, prop_prior_lambda;
  curr_prior_lambda = compute_exp_prior(Lambda);
  prop_prior_lambda = compute_exp_prior(prop_lambda);

  double curr_dens, prop_dens, log_ratio;
  curr_dens = curr_lik + curr_prior_lambda;
  prop_dens = prop_lik + prop_prior_lambda;
  log_ratio = prop_dens - curr_dens;

  if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < log_ratio){
    //ACCEPTED
    std::cout<<"accepted Lambda"<<std::endl;
    acc_sampled_Lambda += 1;
    Lambda.swap(prop_lambda);
  }
  else std::cout<<"rejected Lambda"<<std::endl;

  return;
}

/*
VectorXd UnivariateConditionalMCMC::compute_grad_for_clus(
    int clus, const VectorXd &mean) {
  double grad = 0.0;
  double mean_ = mean(0);
  for (const double datum : data_by_clus[clus])
    grad += (mean_ * (-1) + datum) * a_deltas[clus];

  VectorXd out(1);
  out(0) = grad;
  return out;
}
*/

void UnivariateConditionalMCMC::get_state_as_proto(
    google::protobuf::Message *out_) {
  using namespace google::protobuf::internal;

  UnivariateMixtureState *out = down_cast<UnivariateMixtureState *>(out_);
  out->set_ma(a_means.rows());
  out->set_mna(na_means.rows());
  out->set_mtot(a_means.rows() + na_means.rows());

  to_proto(Map<VectorXd>(a_means.data(), a_means.rows()),
           out->mutable_a_means());
  to_proto(Map<VectorXd>(na_means.data(), na_means.rows()),
           out->mutable_na_means());

  EigenVector *precs = out->mutable_a_deltas();
  precs->set_size(a_deltas.size());
  *precs->mutable_data() = {a_deltas.begin(), a_deltas.end()};

  precs = out->mutable_na_deltas();
  precs->set_size(na_deltas.size());
  *precs->mutable_data() = {na_deltas.begin(), na_deltas.end()};

  to_proto(a_jumps, out->mutable_a_jumps());
  to_proto(na_jumps, out->mutable_na_jumps());

  *out->mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() + ndata};

  out->set_u(u);

  to_proto(Map<VectorXd>(etas.data(), etas.rows()),
           out->mutable_etas());

  to_proto(sigma_bar, out->mutable_sigma_bar());

  LambdaBlock lb;
  lb.set_tau(tau);
  to_proto(Phi, lb.mutable_phi());
  to_proto(Psi, lb.mutable_psi());
  to_proto(Lambda, lb.mutable_lambda());
  out->mutable_lamb_block()->CopyFrom(lb);

  return;

}

void UnivariateConditionalMCMC::print_data_by_clus(int clus) {
  for (const int &d : obs_by_clus[clus]) std::cout << data.row(d) << ", ";
  std::cout << std::endl;
}
};
