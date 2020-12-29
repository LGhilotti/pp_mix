#ifndef CONDITIONAL_MCMC_IMP_HPP
#define CONDITIONAL_MCMC_IMP_HPP


template <class Prec, typename prec_t, typename data_t>
ConditionalMCMC<Prec, prec_t, data_t>::ConditionalMCMC(BasePP *pp_mix,
                                                       BaseJump *h, Prec *g,
                                                       const Params &params)
    : pp_mix(pp_mix), h(h), g(g), params(params) {}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::initialize(
    const std::vector<data_t> &data) {
  this->data = data;
  ndata = data.size();
  set_dim(data[0]);

  MatrixXd ranges = pp_mix->get_ranges(); //USELESS!

  // initialize the allocated means with a "perfect" simulation
  // given an initial number of points
  // int init_n_clus = 5;
  // a_means = pp_mix->sample_n_points(init_n_clus);

  initialize_allocated_means();

  // a_means = MatrixXd::Zero(std::pow(2, dim), dim);
  // for (int i = 0; i < dim; i++) {
  //   int start = 0;
  //   int step = a_means.rows() / (std::pow(2, i + 1));
  //   while (start < a_means.rows()) {
  //     a_means.block(start, i, step, 1) =
  //         MatrixXd::Constant(step, 1, ranges(0, i) / 2.0);
  //     start += step;
  //     a_means.block(start, i, step, 1) =
  //         MatrixXd::Constant(step, 1, ranges(1, i) / 2.0);
  //     start += step;
  //   }
  // }
  std::cout << "a_means: \n" << a_means.transpose() << std::endl;

  // a_means = MatrixXd::Zero(2, dim);
  // a_means.row(0) = ranges.row(0);
  // a_means.row(1) = ranges.row(1);

  // a_means = pp_mix->sample_uniform(5);

  nclus = a_means.rows();
  clus_alloc = VectorXi::Zero(ndata);
  // initial allocation of data into cluster is random
  VectorXd probas = VectorXd::Ones(nclus) / nclus;
  for (int i = 0; i < ndata; i++) {
    clus_alloc(i) = categorical_rng(probas, Rng::Instance().get()) - 1;
  }
  // std::cout << "clus_alloc: " << clus_alloc.transpose() << std::endl;

  // initial vector s(a) has identical element and sums to nclus/(nclus+1)
  a_jumps = VectorXd::Ones(nclus) / (nclus + 1);

  // initial vector Delta(a) is equal to the mean (scalar or matrix)
  a_precs.resize(nclus);
  for (int i = 0; i < nclus; i++) {
    a_precs[i] = g->mean();
    // std::cout << "prec: \n" << a_precs[i] << std::endl;
  }

  // initial mu(na) is just one, uniformly selected in ranges
  na_means = pp_mix->sample_uniform(1);
  // initial s(na) (just one value) is 1/(nclus + 1) -> this way the full s vector sums to 1!
  na_jumps = VectorXd::Ones(na_means.rows()) / (nclus + na_means.rows());

  // initial Delta(na) (just one scalar or matrix) is the mean
  na_precs.resize(na_means.rows());
  for (int i = 0; i < na_means.rows(); i++) {
    na_precs[i] = g->mean();
  }
  // initial u parameter
  u = 1.0;

  std::cout << "initialize done" << std::endl;
}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::run_one() {
  // set T = sum(s1,...,sM)
  double T = a_jumps.sum() + na_jumps.sum();

  // sample u | rest
  u = gamma_rng(ndata, T, Rng::Instance().get());
  // compute laplace transform psi in u
  double psi_u = h->laplace(u);

  // sample c | rest and reorganize the all and nall parameters, and c as well
  sample_allocations_and_relabel();

  // UNALLOCATED PROCESS
  // std::cout << "npoints * " << na_means.rows() << std::endl;
  for (int i=0; i < 10; i++) {
    int npoints = na_means.rows();
    pp_mix->sample_given_active(a_means, &na_means, psi_u);
    if (na_means.rows() != npoints)
      break;
  }
  na_precs.resize(na_means.rows());
  na_jumps.conservativeResize(na_means.rows(), 1);
  for (int i = 0; i < na_means.rows(); i++) {
    na_precs[i] = g->sample_prior();
    na_jumps(i) = h->sample_tilted(u);
  }

  // ALLOCATED PROCES
  // for (int i = 0; i < 10; i++)
  sample_means();
  sample_vars();
  sample_jumps();

  // PERFECT SIMULATION (for updating hypers)
  pp_mix->update_hypers(a_means, na_means);

  // print_debug_string();
}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::sample_allocations_and_relabel() {
  // std::cout << "sample_allocations_and_relabel" << std::endl;
  int Ma = a_means.rows();
  int Mna = na_means.rows();
  // current number of components (a + na)
  int Mtot = Ma + Mna;

  // std::cout << "a_means: \n" << a_means << std::endl;

  const MatrixXd &curr_a_means = a_means;
  const MatrixXd &curr_na_means = na_means;
  const std::vector<prec_t> &curr_a_precs = a_precs;
  const std::vector<prec_t> &curr_na_precs = na_precs;
  const VectorXd &curr_a_jumps = a_jumps;
  const VectorXd &curr_na_jumps = na_jumps;

  // #pragma omp parallel for
  for (int i = 0; i < ndata; i++) {
    VectorXd probas(Mtot);
    // VectorXd mean;
    int newalloc;
    const data_t &datum = data[i];
    probas.head(Ma) = curr_a_jumps;
    probas.tail(Mna) = curr_na_jumps;
    probas = log(probas);

    for (int k = 0; k < Ma; k++) {
      probas[k] += lpdf_given_clus(datum, curr_a_means.row(k).transpose(),
                                   curr_a_precs[k]);
    }
    for (int k = 0; k < Mna; k++) {
      probas[k + Ma] += lpdf_given_clus(datum, curr_na_means.row(k).transpose(),
                                        curr_na_precs[k]);
    }
    // now, we have (probas) the updated probabilities (log and just proportional) for sampling the categorical c_i

    // std::cout << "unnormalized_probas: " << probas.transpose() << std::endl;

    // reconvert probas with the updated probabilities normalized (summing to 1)
    probas = softmax(probas);
    // std::cout << "normalized: " << probas.transpose() << std::endl;
    newalloc = categorical_rng(probas, Rng::Instance().get()) - 1;
    clus_alloc[i] = newalloc;
  }

  _relabel();
}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::_relabel() {
  std::set<int> na2a;  // non active that become active
  std::set<int> a2na;  // active that become non active

  int Ma = a_means.rows();
  int Mna = na_means.rows();
  int Mtot = Ma + Mna;

  for (int i = 0; i < ndata; i++) {
    if (clus_alloc(i) >= Ma) na2a.insert(clus_alloc(i) - Ma);
  }

  // NOW WE RELABEL
  // FIND OUT WHICH CLUSTER HAVE BECOME NON-ACTIVE
  for (int k = 0; k < Ma; k++) {
    if ((clus_alloc.array() == k).count() == 0) a2na.insert(k);
  }
  std::vector<int> a2na_vec(a2na.begin(), a2na.end());
  int n_new_na = a2na.size();
  MatrixXd new_na_means(n_new_na, dim);
  std::vector<prec_t> new_na_precs(n_new_na);
  VectorXd new_na_jumps(n_new_na);

  for (int i = 0; i < n_new_na; i++) {
    new_na_means.row(i) = a_means.row(a2na_vec[i]);
    new_na_precs[i] = a_precs[a2na_vec[i]];
    new_na_jumps(i) = a_jumps(a2na_vec[i]);
  }

  // NOW TAKE CARE OF NON ACTIVE THAT BECOME ACTIVE
  std::vector<int> na2a_vec(na2a.begin(), na2a.end());
  int n_new_a = na2a_vec.size();
  MatrixXd new_a_means(n_new_a, dim);
  std::vector<prec_t> new_a_precs(n_new_a);
  VectorXd new_a_jumps(n_new_a);

  for (int i = 0; i < n_new_a; i++) {
    new_a_means.row(i) = na_means.row(na2a_vec[i]);
    new_a_precs[i] = na_precs[na2a_vec[i]];
    double tmp = na_jumps(na2a_vec[i]);
    new_a_jumps(i) = tmp;
  }

  // delete rows, backward
  for (auto it = a2na_vec.rbegin(); it != a2na_vec.rend(); it++) {
    delete_row(&a_means, *it);
    delete_elem(&a_jumps, *it);
    a_precs.erase(a_precs.begin() + *it);
  }

  // delete rows, backward
  if (na_means.rows() > 0) {
    for (auto it = na2a_vec.rbegin(); it != na2a_vec.rend(); it++) {
      delete_row(&na_means, *it);
      delete_elem(&na_jumps, *it);
      na_precs.erase(na_precs.begin() + *it);
    }
  }

  // NOW JOIN THE STUFF TOGETHER
  if (new_a_means.rows() > 0) {
    int oldMa = a_means.rows();
    a_means.conservativeResize(oldMa + new_a_means.rows(), dim);
    a_means.block(oldMa, 0, new_a_means.rows(), dim) = new_a_means;

    a_jumps.conservativeResize(oldMa + new_a_means.rows());
    a_jumps.segment(oldMa, new_a_means.rows()) = new_a_jumps;

    for (const auto &prec : new_a_precs) a_precs.push_back(prec);
  }

  if (new_na_means.rows() > 0) {
    int oldMna = na_means.rows();
    na_means.conservativeResize(oldMna + new_na_means.rows(), dim);
    na_means.block(oldMna, 0, new_na_means.rows(), dim) = new_na_means;

    na_jumps.conservativeResize(oldMna + new_na_means.rows());
    na_jumps.segment(oldMna, new_na_means.rows()) = new_na_jumps;
  }

  for (const auto &prec : new_na_precs) na_precs.push_back(prec);

  // NOW RELABEL
  std::set<int> uniques_(clus_alloc.data(), clus_alloc.data() + ndata);
  std::vector<int> uniques(uniques_.begin(), uniques_.end());
  std::map<int, int> old2new;
  for (int i = 0; i < uniques.size(); i++) old2new.insert({uniques[i], i});

  for (int i = 0; i < ndata; i++) {
    clus_alloc(i) = old2new[clus_alloc(i)];
  }

  data_by_clus.resize(0);
  data_by_clus.resize(a_means.rows());
  for (int i = 0; i < ndata; i++) {
    data_by_clus[clus_alloc(i)].push_back(data[i]);
  }
}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::sample_means() {

  MatrixXd allmeans(a_means.rows() + na_means.rows(), dim);
  allmeans << a_means, na_means;

  for (int i = 0; i < a_means.rows(); i++) {
    tot_mean += 1;
    MatrixXd others(allmeans.rows() - 1, dim);
    double sigma;
    if (uniform_rng(0, 1, Rng::Instance().get()) < 0.1) {
      sigma = max_proposal_sigma;
    } else
      sigma = min_proposal_sigma;

    double currlik, proplik, prior_ratio, lik_ratio, arate;
    const VectorXd &currmean = a_means.row(i).transpose();
    const MatrixXd &cov_prop = MatrixXd::Identity(dim, dim) * sigma;

    // we PROPOSE a new point from a multivariate normal, with mean equal to the current point
    // and covariance matrix diagonal
    VectorXd prop =
        stan::math::multi_normal_rng(currmean, cov_prop, Rng::Instance().get());

    currlik = lpdf_given_clus_multi(data_by_clus[i], currmean, a_precs[i]);
    proplik = lpdf_given_clus_multi(data_by_clus[i], prop, a_precs[i]);

    lik_ratio = proplik - currlik;
    others = delete_row(allmeans, i);

    prior_ratio =
        pp_mix->papangelou(prop, others) - pp_mix->papangelou(currmean, others);

    arate = lik_ratio + prior_ratio;

    bool accepted = false;
    if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < arate) {
      accepted = true;
      a_means.row(i) = prop.transpose();
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
      // std::cout << "prop_ratio: " << prop_ratio << std::endl;
      std::cout << "ACCEPTED: " << accepted << std::endl;
      std::cout << "**********" << std::endl;
    }
  }
}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::sample_vars() {
#pragma omp parallel for
  for (int i = 0; i < a_means.rows(); i++) {
    a_precs[i] = g->sample_given_data(data_by_clus[i], a_precs[i],
                                      a_means.row(i).transpose());
  }
}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::sample_jumps() {
#pragma omp parallel for
  for (int i = 0; i < a_means.rows(); i++)
    a_jumps(i) = h->sample_given_data(data_by_clus[i].size(), a_jumps(i), u);
}

template <class Prec, typename prec_t, typename data_t>
void ConditionalMCMC<Prec, prec_t, data_t>::print_debug_string() {
  std::cout << "*********** DEBUG STRING***********" << std::endl;
  std::cout << "#### ACTIVE: Number actives: " << a_means.rows() << std::endl;
  ;
  for (int i = 0; i < a_means.rows(); i++) {
    std::cout << "Component: " << i << ", weight: " << a_jumps(i)
              << ", mean: " << a_means.row(i) << std::endl;
    //   << ", precision: " << a_precs[i]
    //   << std::endl;

    std::cout << std::endl;
  }

  std::cout << "#### NON - ACTIVE: Number actives: " << na_means.rows()
            << std::endl;
  ;
  for (int i = 0; i < na_means.rows(); i++) {
    std::cout << "Component: " << i << "weight: " << na_jumps(i)
              << ", mean: " << na_means.row(i) << std::endl;
  }

  std::cout << std::endl;
}

#endif
