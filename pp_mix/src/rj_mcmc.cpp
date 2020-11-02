#include "rj_mcmc.hpp"

#include "rj_mcmc_imp.hpp"

/////////////////////////////////
// UNIVARIATE CONDITIONAL MCMC //
/////////////////////////////////

UnivariateRJMCMC::UnivariateRJMCMC(BasePP *pp_mix, BasePrec *g,
                                   const Params &params)
    : RJMCMC<BaseUnivPrec, double, double>() {
  set_pp_mix(pp_mix);
  set_prec(dynamic_cast<BaseUnivPrec *>(g));
  this->params = params;
  min_proposal_sigma = 0.1;
  max_proposal_sigma = 2.0;
}

void UnivariateRJMCMC::initialize_allocated_means() {
  int init_n_clus = 1;
  if (init_n_clus >= ndata) {
    means.resize(ndata, 1);
    for (int i = 0; i < ndata; i++) means(i, 0) = data[i];
  } else {
    means.resize(init_n_clus, 1);
    std::vector<int> index(ndata);
    std::iota(index.begin(), index.end(), 0);
    std::random_shuffle(index.begin(), index.begin() + ndata);
    for (int i = 0; i < init_n_clus; i++) {
      means(i, 0) = data[index[i]];
    }
  }
}

VectorXd UnivariateRJMCMC::compute_grad_for_clus(int clus,
                                                 const VectorXd &mean) {
  double grad = 0.0;
  double mean_ = mean(0);
  for (const double datum : data_by_clus[clus])
    grad += (mean_ * (-1) + datum) * precs[clus];

  VectorXd out(1);
  out(0) = grad;
  return out;
}

void UnivariateRJMCMC::get_state_as_proto(google::protobuf::Message *out_) {
  using namespace google::protobuf::internal;

  UnivariateMixtureState *out = down_cast<UnivariateMixtureState *>(out_);
  out->set_ma(means.rows());
  out->set_mtot(means.rows());

  to_proto(Map<VectorXd>(means.data(), means.rows()), out->mutable_a_means());

  EigenVector *precsout = out->mutable_a_precs();
  precsout->set_size(precs.size());
  *precsout->mutable_data() = {precs.begin(), precs.end()};

  to_proto(weights, out->mutable_a_jumps());

  *out->mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() + ndata};

  PPState pp_params;
  pp_mix->get_state_as_proto(&pp_params);
  out->mutable_pp_state()->CopyFrom(pp_params);
}

void UnivariateRJMCMC::print_data_by_clus(int clus) {
  for (const double &d : data_by_clus[clus]) std::cout << d << ", ";
  std::cout << std::endl;
}

void UnivariateRJMCMC::combine() {
  // std::cout << "**** COMBINE ****" << std::endl;
  // choose clusters to combine
  VectorXd w_ = VectorXd::Ones(means.rows()) / means.rows();
  int j1 = stan::math::categorical_rng(w_, Rng::Instance().get()) - 1;
  int j2 = j1;
  while (j1 == j2) {
    j2 = stan::math::categorical_rng(w_, Rng::Instance().get()) - 1;
  }
  int j1_ = std::min(j1, j2);
  int j2_ = std::max(j1, j2);
  j1 = j1_;
  j2 = j2_;

  double w_new = weights(j1) + weights(j2);

  double mu_new =
      (weights(j1) * means(j1, 0) + weights(j2) * means(j2, 0)) / w_new;
  double var_new =
      (weights(j1) * (means(j1, 0) * means(j1, 0) + 1.0 / precs[j1]) +
       weights(j2) * (means(j2, 0) * means(j2, 0) + 1.0 / precs[j2])) /
      w_new;
  var_new -= mu_new * mu_new;

  double prec_new = 1.0 / var_new;

  VectorXd new_weights = delete_elem(weights, j2);
  new_weights(j1) = w_new;

  MatrixXd new_means = delete_row(means, j2);
  new_means(j1, 0) = mu_new;
  std::vector<double> new_precs = precs;
  new_precs.erase(new_precs.begin() + j2);

  // compute acceptance probability
  // compute reverse move params
  double alpha = weights(j1) / w_new;
  double r = means(j1, 0) - mu_new;
  r /= std::sqrt(var_new) * std::sqrt(weights(j2) / weights(j1));
  r = std::abs(r);
  double beta =
      (1.0 / precs[j1]) / ((1 - r * r) * w_new / weights(j1) * var_new);

  // log determinant of transformation
  double log_det_J = -std::pow(w_new, 4) /
                     std::pow(weights(j1) * weights(j2), 1.5) *
                     std::pow(var_new, 1.5) * (1 - r * r);

  // likelihood ratio
  double likratio = 0.0;
  VectorXd logweights = log(weights);
  VectorXd lognewweights = log(new_weights);
  for (int i = 0; i < ndata; i++) {
    VectorXd likold = VectorXd::Zero(means.rows());
    VectorXd liknew = VectorXd::Zero(means.rows() - 1);
    for (int j = 0; j < means.rows(); j++) {
      likold(j) =
          logweights(j) + lpdf_given_clus(data[i], means(j, 0), precs[j]);
    }
    for (int j = 0; j < new_means.rows(); j++) {
      liknew(j) = lognewweights(j) +
                  lpdf_given_clus(data[i], new_means(j, 0), new_precs[j]);
    }

    likratio +=
        stan::math::log_sum_exp(liknew) - stan::math::log_sum_exp(likold);
  }

  // prior ratio
  double priorratio =
      g->lpdf(prec_new) - g->lpdf(precs[j1]) - g->lpdf(precs[j2]);

  int n1 = data_by_clus[j1].size();
  int n2 = data_by_clus[j2].size();
  double M = means.rows() - 1;

  priorratio += (prior_dir - 1 + n1 + n2) * std::log(w_new) -
                (prior_dir - 1 + n1) * std::log(weights(j1)) -
                (prior_dir - 1 + n2) * std::log(weights(j2)) +
                stan::math::beta(prior_dir, prior_dir * M);
  // std::cout << "priorratio: " << priorratio << std::endl;

  // std::cout << "means: " << means.transpose() << std::endl;
  // std::cout << "new_means: " << new_means.transpose() << std::endl;

  priorratio += pp_mix->dens(new_means) - pp_mix->dens(means);
  // std::cout << "priorratio: " << priorratio << std::endl;

  double prop_ratio = std::log(0.5 / M * (M + 1.0)) - std::log(0.5 / (M)) -
                      stan::math::beta_lpdf(alpha, 1, 1) -
                      stan::math::beta_lpdf(beta, 1, 1) -
                      stan::math::beta_lpdf(r, 2, 2);

  // std::cout << "combining means: " << means(j1, 0) << ", and " << means(j2, 0)
  //           << ", variances: " << 1.0 / precs[j1] << ", and " << 1.0 / precs[j2]
  //           << ", weights: " << weights(j1) << ", and" << weights(j2) 
  //           << std::endl
  //           << "n1: " << n1 << ", n2: " << n2 << std::endl
  //           << "new_mean: " << mu_new << ", new var: " << var_new << std::endl;
  // std::cout << "likratio: " << likratio << ", priorratio: " << priorratio
  //           << ", prop_ratio: " << prop_ratio << ", log_det_J: " << log_det_J
  //           << std::endl;

  double acc_ratio =
      log_det_J + likratio + priorratio + prop_ratio - std::log(M + 1);

  double u = stan::math::uniform_rng(0, 1, Rng::Instance().get());
  // std::cout << "acc_ratio: " << std::exp(acc_ratio) << std::endl;
  if (std::log(u) < acc_ratio) {
    // std::cout << "ACCEPT" << std::endl;

    weights = new_weights;
    means = new_means;
    precs = new_precs;

    VectorXi new_clus_alloc = clus_alloc;
    for (int i = 0; i < ndata; i++) {
      if (clus_alloc(i) == j2)
        clus_alloc(i) = j1;
      else if (clus_alloc(i) > j2)
        clus_alloc(i) -= 1;
    }
    clus_alloc = new_clus_alloc;
  }
  // std::cout << std::endl << std::endl;
  // std::cout << "done" << std::endl;
}

void UnivariateRJMCMC::split() {
  // std::cout << "**** SPLIT ****" << std::endl;
  int M = means.rows();
  VectorXd w_ = VectorXd::Ones(means.rows()) / means.rows();
  int j = stan::math::categorical_rng(w_, Rng::Instance().get()) - 1;

  double alpha = stan::math::beta_rng(1, 1, Rng::Instance().get());
  double beta = stan::math::beta_rng(1, 1, Rng::Instance().get());
  double r = stan::math::beta_rng(2, 2, Rng::Instance().get());

  double w1_new = alpha * weights(j);
  double w2_new = (1.0 - alpha) * weights(j);

  double std = 1.0 / std::sqrt(precs[j]);

  double m1_new = means(j) - std::sqrt(w2_new / w1_new) * r * std;
  double m2_new = means(j) - std::sqrt(w1_new / w2_new) * r * std;

  double var = 1.0 / precs[j];
  double var1_new = (beta * (1 - r * r) * weights(j) / w1_new) * var;
  double var2_new = ((1 - beta) * (1 - r * r) * weights(j) / w2_new) * var;

  double prec1_new = 1.0 / var1_new;
  double prec2_new = 1.0 / var2_new;

  // new parameters
  VectorXd new_weights(M + 1);
  new_weights.head(M) = weights;
  new_weights(j) = w1_new;
  new_weights(M) = w2_new;

  MatrixXd new_means(M + 1, 1);
  MatrixXd tmp(1, 1);
  tmp(0, 0) = m2_new;
  new_means << means, tmp;
  new_means(j, 0) = m1_new;

  std::vector<double> new_precs = precs;
  new_precs.push_back(prec2_new);
  new_precs[j] = prec1_new;

  // propose new assignments
  VectorXi new_clus_alloc(ndata);
  std::vector<std::vector<double>> new_data_by_clus(M + 1);
  for (int i = 0; i < ndata; i++) {
    if (clus_alloc(i) == j) {
      double p1 =
          w1_new * std::exp(lpdf_given_clus(data[i], m1_new, prec1_new));
      double p2 =
          w2_new * std::exp(lpdf_given_clus(data[i], m2_new, prec2_new));
      if (p1 + p2 == 0) {
        p1 = 0.5;
      } else {
        p1 = p1 / (p1 + p2);
      }
      
      // std::cout << "p1: " << p1 << std::endl;
      int c = stan::math::bernoulli_rng(p1, Rng::Instance().get());
      if (c == 0) {
        new_clus_alloc(i) = M;
      } else {
        new_clus_alloc(i) = j;
      }
    } else {
      new_clus_alloc(i) = clus_alloc(i);
    }
    new_data_by_clus[new_clus_alloc(i)].push_back(data[i]);
  }
  // for (int j = 0; j < M + 1; j++) {
  //   std::cout << "size of cluster: " << j << ": " <<
  //   new_data_by_clus[j].size()
  //             << std::endl;
  // }

  // compute acceptance probability
  // log determinant of transformation
  double log_det_J = std::pow(weights(j), 4) / std::pow(w1_new * w2_new, 1.5) *
                     std::pow(1.0 / precs[j], 1.5) * (1 - r * r);

  // likelihood ratio
  double likratio = 0.0;
  VectorXd logweights = log(weights);
  VectorXd lognewwegiths = log(new_weights);
  for (int i = 0; i < ndata; i++) {
    VectorXd likbig = VectorXd::Zero(M + 1);
    VectorXd liksmall = VectorXd::Zero(M);
    for (int k = 0; k < M + 1; k++) {
      likbig(k) = lognewwegiths(k) +
                  lpdf_given_clus(data[i], new_means(k, 0), new_precs[k]);
    }
    for (int k = 0; k < M; k++) {
      liksmall(k) =
          logweights(k) + lpdf_given_clus(data[i], means(k, 0), precs[k]);
    }

    likratio +=
        stan::math::log_sum_exp(likbig) - stan::math::log_sum_exp(liksmall);
  }

  // std::cout << "prec1: " << prec1_new << ", prec2: " << prec2_new
  //           << ", prec_old: " << precs[j] << std::endl;

  // prior ratio
  double priorratio =
      g->lpdf(prec1_new) + g->lpdf(prec2_new) - g->lpdf(precs[j]);
  // std::cout << "priorratio: " << priorratio << std::endl;

  int n1 = new_data_by_clus[j].size();
  int n2 = new_data_by_clus[M].size();
  // std::cout << "n1: " << n1 << ", n2: " << n2 << std::endl;
  // std::cout << "w1: " << w1_new << ", w2: " << w2_new
  //           << ", w_old: " << weights(j) << std::endl;

  priorratio += (prior_dir - 1 + n1) * std::log(w1_new) +
                (prior_dir - 1 + n2) * std::log(w2_new) -
                (prior_dir - 1 + n1 + n2) * std::log(weights(j)) -
                stan::math::beta(prior_dir, prior_dir * M);
  // std::cout << "priorratio: " << priorratio << std::endl;
  priorratio += pp_mix->dens(new_means) - pp_mix->dens(means);

  double prop_ratio = std::log(0.5 / M * (M + 1.0)) - std::log(0.5 / (M)) -
                      stan::math::beta_lpdf(alpha, 1, 1) -
                      stan::math::beta_lpdf(beta, 1, 1) -
                      stan::math::beta_lpdf(r, 2, 2);

  // std::cout << "splitting mean: " << means(j, 0) << std::endl;
  // std::cout << "likratio: " << likratio << ", priorratio: " << priorratio
  //           << ", prop_ratio: " << prop_ratio << ", log_det_J: " << log_det_J
  //           << std::endl;

  double acc_ratio =
      (log_det_J + likratio + priorratio + prop_ratio - std::log(M + 1));

  double u = stan::math::uniform_rng(0, 1, Rng::Instance().get());
  // std::cout << "acc_ratio: " << std::exp(acc_ratio) << std::endl;
  if (std::log(u) < acc_ratio) {
    // std::cout << "ACCEPT" << std::endl;
    weights = new_weights;
    means = new_means;
    precs = new_precs;
    clus_alloc = new_clus_alloc;
  }
  // std::cout << "done" << std::endl;
}


// MultivariateRJMCMC::MultivariateRJMCMC(BasePP *pp_mix,
//                                        BasePrec *g, const Params &params)
//     : ConditionalMCMC<BaseMultiPrec, PrecMat, VectorXd>() {
//   set_pp_mix(pp_mix);
//   set_prec(dynamic_cast<BaseMultiPrec *>(g));
//   this->params = params;
//   min_proposal_sigma = 0.1;
//   max_proposal_sigma = 2.0;
// }

// void MultivariateRJMCMC::initialize_allocated_means() {
//   int init_n_clus = 1;
//   if (init_n_clus >= ndata) {
//     means.resize(ndata, dim);
//     for (int i = 0; i < ndata; i++) means.row(i) = data[i].transpose();
//   } else {
//     means.resize(init_n_clus, dim);
//     std::vector<int> index(ndata);
//     std::iota(index.begin(), index.end(), 0);
//     std::random_shuffle(index.begin(), index.begin() + ndata);
//     for (int i = 0; i < init_n_clus; i++) {
//       means.row(i) = data[index[i]].transpose();
//     }
//   }
// }

// VectorXd MultivariateRJMCMC::compute_grad_for_clus(int clus,
//                                                    const VectorXd &mean) {
//   VectorXd grad = VectorXd::Zero(dim);
//   for (const VectorXd datum : data_by_clus[clus]) grad += datum - mean;

//   return grad;
// }

// void MultivariateRJMCMC::get_state_as_proto(google::protobuf::Message *out_)
// {
//   using namespace google::protobuf::internal;

//   MultivariateMixtureState *out = down_cast<MultivariateMixtureState
//   *>(out_); out->set_ma(means.rows()); out->set_mtot(means.rows());

//   for (int i = 0; i < a_means.rows(); i++) {
//     EigenVector *mean;
//     EigenMatrix *prec;
//     mean = out->add_a_means();
//     prec = out->add_a_precs();

//     to_proto(means.row(i).transpose(), mean);
//     to_proto(precs[i].get_prec(), prec);
//   }

//   to_proto(weights, out->mutable_a_jumps());
//   *out->mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() +
//   ndata};

//   PPState pp_params;
//   pp_mix->get_state_as_proto(&pp_params);
//   out->mutable_pp_state()->CopyFrom(pp_params);
// }

// void MultivariateRJMCMC::print_data_by_clus(int clus) {
//   for (const VectorXd &d : data_by_clus[clus])
//     std::cout << d.transpose() << std::endl;
// }

// void MultivariateRJMCMC::combine() {
//   using namespace stan::math;
//   auto rng = Rng::Instance().get();

//   VectorXd w_ = VectorXd::Ones(means.rows()) / means.rows();
//   int j1 = categorical_rng(w_, rng) - 1;
//   int j2 = j1;
//   while (j1 == j2) {
//     j2 = categorical_rng(w_, rng) - 1;
//   }
//   int j1_ = std::min(j1, j2);
//   int j2_ = std::max(j1, j2);
//   j1 = j1_;
//   j2 = j2_;

//   VectorXd mean1 = means.row(j1).transpose();
//   VectorXd mean2 = means.row(j2).transpose();
//   MatrixXd var1 = precs[j1].get_var();
//   MatrixXd var2 = precs[j2].get_var();
//   SelfAdjointEigenSolver<MatrixXd> eigensolver1(var1);
//   SelfAdjointEigenSolver<MatrixXd> eigensolver2(var2);
//   VectorXd eigvals1 = eigensolver1.eigenvalues();
//   MarixXd eigvecs1 = eigensolver1.eigenvectors();
//   VectorXd eigvals2 = eigensolver2.eigenvalues();
//   MarixXd eigvecs2 = eigensolver2.eigenvectors();

//   double w_new = weights(j1) + weights(j2);
//   VectorXd mean_new = (mean1 * weights(j1) + mean2 * weights(j2) /  w_new;



// }

// void MultivariateRJMCMC::split() {
//   using namespace stan::math;
//   auto rng = Rng::Instance().get();

//   int M = means.rows();
//   VectorXd w_ = VectorXd::Ones(means.rows()) / means.rows();
//   int j = categorical_rng(w_, Rng::Instance().get()) - 1;
//   VectorXd mean = means.row(j).transpose();
//   MatrixXd var = precs[j].get_var();

//   SelfAdjointEigenSolver<MatrixXd> eigensolver(var);
//   VectorXd eigvals = eigensolver.eigenvalues();
//   MarixXd eigvecs = eigensolver.eigenvectors();

//   double u1 = beta_rng(2, 2, rng);
//   VectorXd u2s(dim);
//   VectorXd u3s(dim);
//   u2s(0) = beta_rng(1, 2*dim, rng);
//   u3s(0) = beta_rng(1, dim, rng);
//   for (int j=1; j < dim; j++) {
//     u2s(j) = uniform_rng(-1, 1, rng);
//     u3s(j) = uniform_rng(-1, 1, rng);
//   }

//   MatrixXd P = MatrixXd::Zero(dim, dim);
//   for (int j=0, j < dim; j++) {
//     for (int k=0; k < j; k++)
//       P(j, k) = uniform_rng(0, 1, rng);
//   }
//   P(j, k) += 
// }