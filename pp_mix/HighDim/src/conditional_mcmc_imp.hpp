#ifndef CONDITIONAL_MCMC_IMP_HPP
#define CONDITIONAL_MCMC_IMP_HPP


template <class Prec, typename prec_t, typename fact_t>
ConditionalMCMC<Prec, prec_t, fact_t>::ConditionalMCMC(BaseDeterminantalPP *pp_mix,
                                                  Prec *g,
                                                  const Params &params)
    : pp_mix(pp_mix), g(g) {

      set_params(params);
      return;
    }

template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::set_params(const Params & p){

  this->params = p;
  set_dim_factor();
  this->_a_phi = params.a();
  this->_alpha_jump = params.alphajump();
  this->_beta_jump = params.betajump();
  this->_a_gamma = params.agamma();
  this->_b_gamma = params.bgamma();
  this->prop_lambda_sigma = params.prop_sigma();
  return;
}

template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::initialize(
    const MatrixXd &dat) {

  this->data = dat;
  ndata = data.rows();
  dim_data = data.cols();

  // Initialize Lambda block: tau, psi, phi and Lambda
  tau = 2.0 * dim_data * dim_fact * _a_phi ;
  Phi = 1.0/(dim_data*dim_fact) * MatrixXi::Ones(dim_data,dim_fact) ;
  Psi = 2.0 * MatrixXi::Ones(dim_data,dim_fact);

  Lambda = Map<MatrixXd>(normal_rng(VectorXd::Zero(dim_data*dim_fact),
        8.*pow(_a_phi,2)*VectorXd::Ones(dim_data*dim_fact)).data() , dim_data,dim_fact );

  // Initialize Sigma_bar
  sigma_bar = _a_gamma/_b_gamma * VectorXd::Ones(dim_data);

  // Initialize etas
  initialize_etas(dat);

  // REP-PP BLOCK
  // Initialize the allocated means
  initialize_allocated_means();
  std::cout << "a_means: \n" << a_means.transpose() << std::endl;

  double nclus = a_means.rows();
  clus_alloc = VectorXi::Zero(ndata);
  // Initialize cluster allocations
  // Initial allocation of data into cluster is random: some a_means could have
  // zero elements associated.
  VectorXd probas = VectorXd::Ones(nclus) / nclus;
  for (int i = 0; i < ndata; i++) {
    clus_alloc(i) = categorical_rng(probas, Rng::Instance().get()) - 1;
  }

  // initial vector s(a) has identical element and sums to nclus/(nclus+1)
  a_jumps = VectorXd::Ones(nclus) / (nclus + 1);

  // initial vector Delta(a) is equal to the mean (scalar or matrix)
  a_deltas.resize(nclus);
  for (int i = 0; i < nclus; i++) {
    a_deltas[i] = g->mean();
    // std::cout << "prec: \n" << a_deltas[i] << std::endl;
  }

  // initial mu(na) is just one, uniformly selected in ranges
  na_means = pp_mix->sample_uniform(1);
  // initial s(na) (just one value) is 1/(nclus + 1) -> this way the full s vector sums to 1!
  na_jumps = VectorXd::Ones(na_means.rows()) / (nclus + na_means.rows());

  // initial Delta(na) (just one scalar or matrix) is the mean
  na_deltas.resize(na_means.rows());
  for (int i = 0; i < na_means.rows(); i++) {
    na_deltas[i] = g->mean();
  }
  // initial u parameter
  u = 1.0;

  // DECOMPOSE DPP (in MultiDpp also assign the pointer to Lambda)
  pp_mix->set_decomposition(&Lambda);

  std::cout << "initialize done" << std::endl;
}

template <class Prec, typename prec_t, typename fact_t>
std::vector<VectorXd> ConditionalMCMC<Prec, prec_t, fact_t>::proj_inside() {

  std::vector<VectorXd> inside;
  for (int i=0; i<ndata;i++){
    if (is_inside(etas.row(i))){
      inside.push_back(etas.row(i));
    }
  }
  return inside;
}

template <class Prec, typename prec_t, typename fact_t>
bool ConditionalMCMC<Prec, prec_t, fact_t>::is_inside(const VectorXd & eta){
  MatrixXd ran = pp_mix->get_ranges();
  bool is_in = true;
  for (int i=0;i<dim_fact&& is_in==true;i++){
    if (eta(i)<ran(0,i) || eta(i)>ran(1,i)){
      is_in = false;
    }
  }
  return is_in;
}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::run_one() {

  std::cout<< "begin iteration"<<std::endl;

  sample_u();

  // compute laplace transform psi in u
  double psi_u = laplace(u);

  // sample c | rest and reorganize the all and nall parameters, and c as well
  sample_allocations_and_relabel();

  // sample non-allocated variables
  sample_means_na();

  sample_jumps_na();

  sample_deltas_na();

  // sample allocated variables
  sample_means_a();

  sample_deltas_a();

  sample_jumps_a();

  // sample etas
  sample_etas();

  // sample Sigma bar
  sample_sigma_bar();

  // sample Lambda block
  sample_Psi();
  sample_tau();
  sample_Phi();
  sample_Lambda();

  std::cout<<"end iteration"<<std::endl;
  // print_debug_string();
  return;
}

template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_jumps_na()
{
    na_jumps.conservativeResize(na_means.rows(), 1);
    int N_na = na_jumps.size();
    na_jumps = gamma_rng(_alpha_jump * VectorXd::Ones(N_na),
            (_beta_jump + u) * VectorXd::Ones(N_na) , Rng::Instance().get());
    return;
}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_jumps_a()
{
    //#pragma omp parallel for
    for (int i = 0; i < a_means.rows(); i++)
      a_jumps(i) = gamma_rng(_alpha_jump+etas_by_clus[i].size(), _beta_jump+u , Rng::Instance().get());

    return;
}

template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_means_na()
{
  for (int i=0; i < 10; i++) {
    int na_points = na_means.rows();
    pp_mix->sample_nonalloc_fullcond(&na_means, a_means, psi_u);
    if (na_means.rows() != na_points)
      break;
  }
  return;
}

template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_means_a()
{
  MatrixXd allmeans(a_means.rows() + na_means.rows(), dim_fact);
  allmeans << a_means, na_means;

  for (int i = 0; i < a_means.rows(); i++) {
    tot_mean += 1;
    MatrixXd others(allmeans.rows() - 1, dim_fact);
    double sigma;
    if (uniform_rng(0, 1, Rng::Instance().get()) < 0.1) {
      sigma = max_proposal_sigma;
    } else
      sigma = min_proposal_sigma;

    double currlik, proplik, prior_ratio, lik_ratio, arate;
    const VectorXd &currmean = a_means.row(i).transpose();
    const MatrixXd &cov_prop = MatrixXd::Identity(dim_fact, dim_fact) * sigma;

    // we PROPOSE a new point from a multivariate normal, with mean equal to the current point
    // and covariance matrix diagonal
    VectorXd prop =
        stan::math::multi_normal_rng(currmean, cov_prop, Rng::Instance().get());

    currlik = lpdf_given_clus_multi(etas_by_clus[i], currmean, a_deltas[i]);
    proplik = lpdf_given_clus_multi(etas_by_clus[i], prop, a_deltas[i]);

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


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_deltas_na() {

  na_deltas.resize(na_means.rows());
  for (int i = 0; i < na_means.rows(); i++) {
    na_deltas[i] = g->sample_prior();
  }
  return;
}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_deltas_a() {

  //#pragma omp parallel for
  for (int i = 0; i < a_means.rows(); i++) {
    a_deltas[i] = g->sample_alloc(etas_by_clus[i], a_deltas[i],
                                      a_means.row(i).transpose());
  }
  return;
}



template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_allocations_and_relabel() {
  std::cout << "sample_allocations_and_relabel" << std::endl;
  int Ma = a_means.rows();
  int Mna = na_means.rows();
  // current number of components (a + na)
  int Mtot = Ma + Mna;

  // std::cout << "a_means: \n" << a_means << std::endl;

  const MatrixXd &curr_a_means = a_means;
  const MatrixXd &curr_na_means = na_means;
  const std::vector<prec_t, fact_t> &curr_a_deltas = a_deltas;
  const std::vector<prec_t, fact_t> &curr_na_deltas = na_deltas;
  const VectorXd &curr_a_jumps = a_jumps;
  const VectorXd &curr_na_jumps = na_jumps;

  // #pragma omp parallel for
  for (int i = 0; i < ndata; i++) {
    VectorXd probas(Mtot);
    // VectorXd mean;
    int newalloc;
    const VectorXd &eta = etas.row(i).transpose();
    probas.head(Ma) = curr_a_jumps;
    probas.tail(Mna) = curr_na_jumps;
    probas = log(probas);

    for (int k = 0; k < Ma; k++) {
      probas[k] += lpdf_given_clus(eta, curr_a_means.row(k).transpose(),
                                   curr_a_deltas[k]);
    }
    for (int k = 0; k < Mna; k++) {
      probas[k + Ma] += lpdf_given_clus(eta, curr_na_means.row(k).transpose(),
                                        curr_na_deltas[k]);
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

template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::_relabel() {
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
  MatrixXd new_na_means(n_new_na, dim_fact);
  std::vector<prec_t, fact_t> new_na_deltas(n_new_na);
  VectorXd new_na_jumps(n_new_na);

  for (int i = 0; i < n_new_na; i++) {
    new_na_means.row(i) = a_means.row(a2na_vec[i]);
    new_na_deltas[i] = a_deltas[a2na_vec[i]];
    new_na_jumps(i) = a_jumps(a2na_vec[i]);
  }

  // NOW TAKE CARE OF NON ACTIVE THAT BECOME ACTIVE
  std::vector<int> na2a_vec(na2a.begin(), na2a.end());
  int n_new_a = na2a_vec.size();
  MatrixXd new_a_means(n_new_a, dim_fact);
  std::vector<prec_t, fact_t> new_a_deltas(n_new_a);
  VectorXd new_a_jumps(n_new_a);

  for (int i = 0; i < n_new_a; i++) {
    new_a_means.row(i) = na_means.row(na2a_vec[i]);
    new_a_deltas[i] = na_deltas[na2a_vec[i]];
    double tmp = na_jumps(na2a_vec[i]);
    new_a_jumps(i) = tmp;
  }

  // delete rows, backward
  for (auto it = a2na_vec.rbegin(); it != a2na_vec.rend(); it++) {
    delete_row(&a_means, *it);
    delete_elem(&a_jumps, *it);
    a_deltas.erase(a_deltas.begin() + *it);
  }

  // delete rows, backward
  if (na_means.rows() > 0) {
    for (auto it = na2a_vec.rbegin(); it != na2a_vec.rend(); it++) {
      delete_row(&na_means, *it);
      delete_elem(&na_jumps, *it);
      na_deltas.erase(na_deltas.begin() + *it);
    }
  }

  // NOW JOIN THE STUFF TOGETHER
  if (new_a_means.rows() > 0) {
    int oldMa = a_means.rows();
    a_means.conservativeResize(oldMa + new_a_means.rows(), dim_fact);
    a_means.block(oldMa, 0, new_a_means.rows(), dim_fact) = new_a_means;

    a_jumps.conservativeResize(oldMa + new_a_means.rows());
    a_jumps.segment(oldMa, new_a_means.rows()) = new_a_jumps;

    for (const auto &prec : new_a_deltas) a_deltas.push_back(prec);
  }

  if (new_na_means.rows() > 0) {
    int oldMna = na_means.rows();
    na_means.conservativeResize(oldMna + new_na_means.rows(), dim_fact);
    na_means.block(oldMna, 0, new_na_means.rows(), dim_fact) = new_na_means;

    na_jumps.conservativeResize(oldMna + new_na_means.rows());
    na_jumps.segment(oldMna, new_na_means.rows()) = new_na_jumps;
  }

  for (const auto &prec : new_na_deltas) na_deltas.push_back(prec);

  // NOW RELABEL
  std::set<int> uniques_(clus_alloc.data(), clus_alloc.data() + ndata);
  std::vector<int> uniques(uniques_.begin(), uniques_.end());
  std::map<int, int> old2new;
  for (int i = 0; i < uniques.size(); i++) old2new.insert({uniques[i], i});

  for (int i = 0; i < ndata; i++) {
    clus_alloc(i) = old2new[clus_alloc(i)];
  }

  obs_by_clus.resize(0);
  obs_by_clus.resize(a_means.rows());
  etas_by_clus.resize(0);
  etas_by_clus.resize(a_means.rows());
  for (int i = 0; i < ndata; i++) {
    obs_by_clus[clus_alloc(i)].push_back(i);
    etas_by_clus[clus_alloc(i)].push_back(etas.row(i).transpose());
  }
}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_sigma_bar() {

  VectorXd alphas = VectorXd::Constant(dim_data, ndata/2.0 + _a_gamma);

  VectorXd betas = VectorXd::Constant(dim_data, _b_gamma);
  betas+=0.5*data.colwise().squaredNorm().transpose() + 0.5*(etas*Lambda.transpose()).colwise().squaredNorm().transpose();
  for (int j=0; j < dim_data; j++){
    betas(j)-= Lambda.row(j)*etas.transpose()*data.col(j);
  }

  sigma_bar = gamma_rng(alphas,betas,Rng::Instance().get());

  return;

}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_Psi() {
  // make it parallel omp
  for (int j=0; j< dim_data; j++)
    for(int h=0; h< dim_fact; h++)
      Psi(j,h)=GIG::rgig(0.5, std::pow(Lambda(j,h)/(Phi(j,h)*tau),2), 1);

  return;
}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_tau() {

  tau = GIG::rgig(dim_data*dim_fact*(_a_phi-1), 2*(Lambda.array()/Phi.array()).sum() , 1);
  return;
}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::sample_Phi() {

  for (int j=0; j < dim_data; j++)
    for (int h=0; h < dim_fact; h++)
      Phi(j,h)=GIG:rgig( _a_phi-1 , 2.0*std::abs(Lambda(j,h)), 1);

  Phi/= Phi.sum();
  return;
}



template <class Prec, typename prec_t, typename fact_t>
double ConditionalMCMC<Prec, prec_t, fact_t>::compute_exp_lik(const MatrixXd& lamb) const {

  return (sigma_bar.array().sqrt().matrix().asDiagonal()*(data.transpose() - lamb*etas.transpose())).colwise().squaredNorm().sum();

}

template <class Prec, typename prec_t, typename fact_t>
double ConditionalMCMC<Prec, prec_t, fact_t>::compute_exp_prior(const MatrixXd& lamb) const {

  return (lamb.array().square()/(Psi.array()*Phi.array().square())).sum() * (- 0.5 / (tau*tau));

}


template <class Prec, typename prec_t, typename fact_t>
void ConditionalMCMC<Prec, prec_t, fact_t>::print_debug_string() {
  std::cout << "*********** DEBUG STRING***********" << std::endl;
  std::cout << "#### ACTIVE: Number actives: " << a_means.rows() << std::endl;
  ;
  for (int i = 0; i < a_means.rows(); i++) {
    std::cout << "Component: " << i << ", weight: " << a_jumps(i)
              << ", mean: " << a_means.row(i) << std::endl;
    //   << ", precision: " << a_deltas[i]
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
