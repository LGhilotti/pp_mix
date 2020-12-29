#ifndef RJ_MCMC_IMP_HPP
#define RJ_MCMC_IMP_HPP

template <class Prec, typename prec_t, typename data_t>
RJMCMC<Prec, prec_t, data_t>::RJMCMC(BasePP *pp_mix, Prec *g,
                                     const Params &params)
    : pp_mix(pp_mix), g(g), params(params) {}

template <class Prec, typename prec_t, typename data_t>
void RJMCMC<Prec, prec_t, data_t>::initialize(const std::vector<data_t> &data) {
  this->data = data;
  ndata = data.size();
  set_dim(data[0]);

  MatrixXd ranges = pp_mix->get_ranges();

  initialize_allocated_means();

  std::cout << "means: \n" << means.transpose() << std::endl;

  nclus = means.rows();
  clus_alloc = VectorXi::Zero(ndata);
  VectorXd probas = VectorXd::Ones(nclus) / nclus;
  for (int i = 0; i < ndata; i++) {
    clus_alloc(i) = categorical_rng(probas, Rng::Instance().get()) - 1;
  }
  // std::cout << "clus_alloc: " << clus_alloc.transpose() << std::endl;

  weights = VectorXd::Ones(nclus) / (nclus);

  precs.resize(nclus);
  for (int i = 0; i < nclus; i++) {
    precs[i] = g->mean();
  }

  sample_allocations();

  std::cout << "initialize done" << std::endl;
}

template <class Prec, typename prec_t, typename data_t>
void RJMCMC<Prec, prec_t, data_t>::run_one() {
  // std::cout << "allocations done" << std::endl;
  sample_means();
  // std::cout << "means done" << std::endl;
  sample_vars();
  // std::cout << "vars done" << std::endl;
  sample_weights();
  // std::cout << "weights done" << std::endl;

  if (means.rows() == 1)
    split();
  else {
    if (stan::math::uniform_rng(0, 1, Rng::Instance().get()) < 0.5)
      split();
    else
      combine();
  }
  
  sample_allocations();
}

template <class Prec, typename prec_t, typename data_t>
void RJMCMC<Prec, prec_t, data_t>::sample_allocations() {
  // #pragma omp parallel for
  for (int i = 0; i < ndata; i++) {
    VectorXd probas(means.rows());
    // VectorXd mean;
    int newalloc;
    const data_t &datum = data[i];
    probas = log(weights);

    for (int k = 0; k < means.rows(); k++) {
      probas[k] += lpdf_given_clus(datum, means.row(k).transpose(), precs[k]);
    }
    probas = softmax(probas);
    newalloc = categorical_rng(probas, Rng::Instance().get()) - 1;
    clus_alloc[i] = newalloc;
  }

  data_by_clus.resize(0);
  data_by_clus.resize(means.rows());
  for (int i = 0; i < ndata; i++) {
    data_by_clus[clus_alloc(i)].push_back(data[i]);
  }
}

template <class Prec, typename prec_t, typename data_t>
void RJMCMC<Prec, prec_t, data_t>::sample_means() {
  const MatrixXd &ranges = pp_mix->get_ranges();
  // We update each mean separately

  for (int i = 0; i < means.rows(); i++) {
    tot_mean += 1;
    MatrixXd others(means.rows() - 1, dim);
    double sigma;
    if (uniform_rng(0, 1, Rng::Instance().get()) < 0.1) {
      sigma = max_proposal_sigma;
    } else
      sigma = min_proposal_sigma;

    double stepsize = params.mala_stepsize();
    double currlik, proplik, prior_ratio, lik_ratio, arate;
    const VectorXd &currmean = means.row(i).transpose();
    const MatrixXd &cov_prop = MatrixXd::Identity(dim, dim) * sigma;

    VectorXd prop =
        stan::math::multi_normal_rng(currmean, cov_prop, Rng::Instance().get());

    currlik = lpdf_given_clus_multi(data_by_clus[i], currmean, precs[i]);
    proplik = lpdf_given_clus_multi(data_by_clus[i], prop, precs[i]);

    lik_ratio = proplik - currlik;
    others = delete_row(means, i);

    prior_ratio =
        pp_mix->papangelou(prop, others) - pp_mix->papangelou(currmean, others);

    arate = lik_ratio + prior_ratio;

    bool accepted = false;
    if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < arate) {
      accepted = true;
      means.row(i) = prop.transpose();
      acc_mean += 1;
    }

    if (verbose) {
      std::cout << "Component: " << i << std::endl;
      std::cout << "data:" << std::endl;
      print_data_by_clus(i);
      std::cout << "currmean: " << currmean.transpose()
                << ", currlik: " << currlik << std::endl;
      std::cout << "prop: " << prop.transpose() << ", proplik: " << proplik
                << std::endl;
      std::cout << "prior_ratio: " << prior_ratio << std::endl;
      std::cout << "prop_papangelou: " << pp_mix->papangelou(prop, others)
                << ", curr_papangelou: " << pp_mix->papangelou(currmean, others)
                << std::endl;
      std::cout << "lik_ratio: " << lik_ratio << std::endl;
      std::cout << "ACCEPTED: " << accepted << std::endl;
      std::cout << "**********" << std::endl;
    }
  }
}

template <class Prec, typename prec_t, typename data_t>
void RJMCMC<Prec, prec_t, data_t>::sample_vars() {
#pragma omp parallel for
  for (int i = 0; i < means.rows(); i++) {
    precs[i] = g->sample_given_data(data_by_clus[i], precs[i],
                                    means.row(i).transpose());
  }
}

template <class Prec, typename prec_t, typename data_t>
void RJMCMC<Prec, prec_t, data_t>::sample_weights() {
  VectorXd params_dir(means.rows());
  for (int i = 0; i < means.rows(); i++)
    params_dir(i) = prior_dir + data_by_clus[i].size();

  weights = stan::math::dirichlet_rng(params_dir, Rng::Instance().get());
}

template <class Prec, typename prec_t, typename data_t>
void RJMCMC<Prec, prec_t, data_t>::print_debug_string() {
  std::cout << "*********** DEBUG STRING***********" << std::endl;
  std::cout << "#### ACTIVE: Number actives: " << means.rows() << std::endl;
  ;
  for (int i = 0; i < means.rows(); i++) {
    std::cout << "Component: " << i << ", weight: " << weights(i)
              << ", mean: " << means.row(i) << ", prec: " << precs[i]
              << std::endl;
    std::cout << std::endl;
  }

  std::cout << std::endl;
}

#endif