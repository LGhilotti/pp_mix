#include "conditional_mcmc.hpp"

#include "conditional_mcmc_imp.hpp"

MultivariateConditionalMCMC::MultivariateConditionalMCMC(BasePP *pp_mix,
                                                         BaseJump *h,
                                                         BasePrec *g,
                                                         const Params &params)
    : ConditionalMCMC<BaseMultiPrec, PrecMat, VectorXd>() {
  set_pp_mix(pp_mix);
  set_jump(h);
  set_prec(dynamic_cast<BaseMultiPrec *>(g));
  this->params = params;
  min_proposal_sigma = 0.1;
  max_proposal_sigma = 2.0;
}

void MultivariateConditionalMCMC::initialize_allocated_means() {
  int init_n_clus = 10;
  if (init_n_clus >= ndata) {
    a_means.resize(ndata, dim);
    for (int i=0; i < ndata; i++) 
      a_means.row(i) = data[i].transpose();
  } else {
    a_means.resize(init_n_clus, dim);
    std::vector<int> index(ndata);
    std::iota(index.begin(), index.end(), 0);
    std::random_shuffle(index.begin(), index.begin() + ndata);
    for (int i=0; i < init_n_clus; i++) {
      a_means.row(i) = data[index[i]].transpose();
    }
  }
}

VectorXd MultivariateConditionalMCMC::compute_grad_for_clus(
    int clus, const VectorXd &mean) {
  VectorXd grad = VectorXd::Zero(dim);
  for (const VectorXd datum : data_by_clus[clus]) grad += datum - mean;
  // TODO FIXME
  // grad += a_precs[clus].get_prec() * (datum - mean);

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
    prec = out->add_a_precs();

    to_proto(a_means.row(i).transpose(), mean);
    to_proto(a_precs[i].get_prec(), prec);
  }

  for (int i = 0; i < na_means.rows(); i++) {
    EigenVector *mean;
    EigenMatrix *prec;
    mean = out->add_na_means();
    prec = out->add_na_precs();

    to_proto(na_means.row(i).transpose(), mean);
    to_proto(na_precs[i].get_prec(), prec);
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
  for (const VectorXd &d : data_by_clus[clus])
    std::cout << d.transpose() << std::endl;
}

/////////////////////////////////
// UNIVARIATE CONDITIONAL MCMC //
/////////////////////////////////

UnivariateConditionalMCMC::UnivariateConditionalMCMC(BasePP *pp_mix,
                                                     BaseJump *h, BasePrec *g,
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
    for (int i = 0; i < ndata; i++) a_means(i, 0) = data[i];
  } else {
    a_means.resize(init_n_clus, 1);
    std::vector<int> index(ndata);
    std::iota(index.begin(), index.end(), 0);
    std::random_shuffle(index.begin(), index.begin() + ndata);
    for (int i = 0; i < init_n_clus; i++) {
      a_means(i, 0) = data[index[i]];
    }
  }
}

VectorXd UnivariateConditionalMCMC::compute_grad_for_clus(
    int clus, const VectorXd &mean) {
  double grad = 0.0;
  double mean_ = mean(0);
  for (const double datum : data_by_clus[clus])
    grad += (mean_ * (-1) + datum) * a_precs[clus];

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

  EigenVector *precs = out->mutable_a_precs();
  precs->set_size(a_precs.size());
  *precs->mutable_data() = {a_precs.begin(), a_precs.end()};

  precs = out->mutable_na_precs();
  precs->set_size(na_precs.size());
  *precs->mutable_data() = {na_precs.begin(), na_precs.end()};

  to_proto(a_jumps, out->mutable_a_jumps());
  to_proto(na_jumps, out->mutable_na_jumps());

  *out->mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() + ndata};

  out->set_u(u);

  PPState pp_params;
  pp_mix->get_state_as_proto(&pp_params);
  out->mutable_pp_state()->CopyFrom(pp_params);
}

void UnivariateConditionalMCMC::print_data_by_clus(int clus) {
  for (const double &d : data_by_clus[clus]) std::cout << d << ", ";
  std::cout << std::endl;
}

///////////////////////////////
// BERNOULI CONDITIONAL MCMC //
///////////////////////////////

BernoulliConditionalMCMC::BernoulliConditionalMCMC(BasePP *pp_mix, BaseJump *h,
                                                   BasePrec *g,
                                                   const Params &params) {
  set_pp_mix(pp_mix);
  set_jump(h);
  set_prec(dynamic_cast<FakePrec *>(g));
  this->params = params;
  min_proposal_sigma = 0.025;
  max_proposal_sigma = 0.3;
}

void BernoulliConditionalMCMC::initialize_allocated_means() {
  int init_n_clus = 5;
  a_means = pp_mix->sample_n_points(init_n_clus);
}

double BernoulliConditionalMCMC::lpdf_given_clus(const VectorXd &x,
                                                 const VectorXd &mu,
                                                 const PrecMat &sigma) {
//   std::cout << "x: " << x.transpose() << std::endl;
//   std::cout << "mu: " << mu.transpose() << std::endl;

  return (x.array() * mu.array().log()).sum() +
         ((1.0 - x.array()) * (1.0 - mu.array()).log()).sum();
}

double BernoulliConditionalMCMC::lpdf_given_clus_multi(
    const std::vector<VectorXd> &x, const VectorXd &mu, const PrecMat &sigma) {
  double out =
      std::accumulate(x.begin(), x.end(), 0.0,
                      [this, &mu, &sigma](double tot, const VectorXd &curr) {
                        return tot + lpdf_given_clus(curr, mu, sigma);
                      });
  return out;
}

VectorXd BernoulliConditionalMCMC::compute_grad_for_clus(int clus,
                                                         const VectorXd &mean) {
  VectorXd grad = VectorXd::Zero(dim);
  // ArrayXd meanarr = mean.array();
  // for (const VectorXd& datum : data_by_clus[clus]) {
  //   ArrayXd tmp =
  //       datum.array() / meanarr - (1.0 - datum.array()) / (1.0 - meanarr);
  //   grad += tmp.matrix();
  // }

  return grad;
}

void BernoulliConditionalMCMC::get_state_as_proto(
    google::protobuf::Message *out_) {
  using namespace google::protobuf::internal;

  BernoulliMixtureState *out = down_cast<BernoulliMixtureState *>(out_);
  out->set_ma(a_means.rows());
  out->set_mna(na_means.rows());
  out->set_mtot(a_means.rows() + na_means.rows());

  for (int i = 0; i < a_means.rows(); i++) {
    EigenVector *mean;
    mean = out->add_a_probs();
    to_proto(a_means.row(i).transpose(), mean);
  }

  for (int i = 0; i < na_means.rows(); i++) {
    EigenVector *mean;
    mean = out->add_na_probs();
    to_proto(na_means.row(i).transpose(), mean);
  }

  to_proto(a_jumps, out->mutable_a_jumps());
  to_proto(na_jumps, out->mutable_na_jumps());

  *out->mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() + ndata};

  out->set_u(u);

  PPState pp_params;
  pp_mix->get_state_as_proto(&pp_params);
  out->mutable_pp_state()->CopyFrom(pp_params);
}

void BernoulliConditionalMCMC::print_data_by_clus(int clus) {
  for (const VectorXd &d : data_by_clus[clus])
    std::cout << d.transpose() << std::endl;
}