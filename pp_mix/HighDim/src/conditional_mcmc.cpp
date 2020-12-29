#include "conditional_mcmc.hpp"

#include "conditional_mcmc_imp.hpp"

MultivariateConditionalMCMC::MultivariateConditionalMCMC(BaseDeterminantalPP *pp_mix,
                                                         GammaJump *h,
                                                         BasePrec *g,
                                                         const Params &params)
    : ConditionalMCMC<BaseMultiPrec, PrecMat>() {
  set_pp_mix(pp_mix);
  set_jump(h);
  set_prec(dynamic_cast<BaseMultiPrec *>(g));
  set_params(params);
  min_proposal_sigma = 0.1;
  max_proposal_sigma = 2.0;
}

void MultivariateConditionalMCMC::initialize_etas(const MatrixXd &dat) {

  LLT<MatrixXd> M (Lambda.transpose() * Lambda);

  etas = (M.solve((dat*Lambda).transpose())).transpose();

  return;

}

std::vector<VectorXd> MultivariateConditionalMCMC::proj_inside() {

  std::vector<VectorXd> inside;
  for (int i=0; i<ndata;i++){
    if (is_inside(etas.row(i))){
      inside.push_back(etas.row(i));
    }
  }
  return inside;
}


bool MultivariateConditionalMCMC::is_inside(const VectorXd & eta){
  MatrixXd ran = pp_mix->get_ranges();
  bool is_in = true;
  for (int i=0;i<dim_fact&& is_in==true;i++){
    if (eta(i)<ran(0,i) || eta(i)>ran(1,i)){
      is_in = false;
    }
  }
  return is_in;
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
      MatrixXd B1(dim_fact,obs_by_clus[i].size());
      B1 = (a_deltas[i].get_prec() * a_means.row(i).transpose()).replicate(1,B1.cols());
      B1 +=M2(all,obs_by_clus[i]);
      // each modified row has solution for points in the cluster.
      G(obs_by_clus[i],all)=(Sn_bar_cho[i].solve(B1)).transpose();
    }

    // Here, G contains (in each row) mean of full-cond, while precisions have to be taken from Sn_bar
    // Now, I sample each eta from the full-cond multi-normal
    for (int i=0; i < ndata; i++){
      etas.row(i)=multi_normal_prec_rng(G.row(i).transpose(), Sn_bar[clus_alloc(i)], Rng::Instance().get());
    }
    return;
}


void MultivariateConditionalMCMC::sample_Lambda() {

  VectorXd prop_lambda_vec = normal_rng(Map<VectorXd>(Lambda.data(), dim_data*dim_fact),
              VectorXd::Constant(dim_data*dim_fact, prop_lambda_sigma));

  MatrixXd prop_lambda = Map<MatrixXd>(prop_lambda_vec.data(),dim_data,dim_fact);

  double curr_lik, prop_lik;
  curr_lik = std::exp(-0.5 * compute_exp_lik(Lambda));
  prop_lik = std::exp(-0.5 * compute_exp_lik(prop_lambda));

  double curr_prior_cond_process, prop_prior_cond_process;
  MatrixXd means(a_means.rows()+na_means.rows(),dim_fact);
  means << a_means, na_means;

  curr_prior_cond_process = pp_mix->dens_cond_process()

  return;
}


VectorXd MultivariateConditionalMCMC::compute_grad_for_clus(
    int clus, const VectorXd &mean) {
  VectorXd grad = VectorXd::Zero(dim);
  for (const VectorXd datum : data_by_clus[clus]) grad += datum - mean;
  // TODO FIXME
  // grad += a_deltas[clus].get_prec() * (datum - mean);

  return grad;
}

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

  PPState pp_params;
  pp_mix->get_state_as_proto(&pp_params);
  out->mutable_pp_state()->CopyFrom(pp_params);
}

void MultivariateConditionalMCMC::print_data_by_clus(int clus) {
  for (const int &d : obs_by_clus[clus])
    std::cout << data.row(d).transpose() << std::endl;
}

/////////////////////////////////
// UNIVARIATE CONDITIONAL MCMC //
/////////////////////////////////

UnivariateConditionalMCMC::UnivariateConditionalMCMC(BaseDeterminantalPP *pp_mix,
                                                     GammaJump *h, BasePrec *g,
                                                     const Params &params)
    : ConditionalMCMC<BaseUnivPrec, double, double>() {
  set_pp_mix(pp_mix);
  set_jump(h);
  set_prec(dynamic_cast<BaseUnivPrec *>(g));
  this->params = params;
  min_proposal_sigma = 0.1;
  max_proposal_sigma = 1.0;
}

void UnivariateConditionalMCMC::initialize_allocated_means() {
  int init_n_clus = 4;
  if (init_n_clus >= ndata) {
    a_means.resize(ndata, 1);
    for (int i = 0; i < ndata; i++) a_means(i, 0) = data.row(i);
  } else {
    a_means.resize(init_n_clus, 1);
    std::vector<int> index(ndata);
    std::iota(index.begin(), index.end(), 0);
    std::random_shuffle(index.begin(), index.begin() + ndata);
    for (int i = 0; i < init_n_clus; i++) {
      a_means(i, 0) = data.row(index[i]);
    }
  }
}

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

  PPState pp_params;
  pp_mix->get_state_as_proto(&pp_params);
  out->mutable_pp_state()->CopyFrom(pp_params);
}

void UnivariateConditionalMCMC::print_data_by_clus(int clus) {
  for (const int &d : obs_by_clus[clus]) std::cout << data.row(d) << ", ";
  std::cout << std::endl;
}
