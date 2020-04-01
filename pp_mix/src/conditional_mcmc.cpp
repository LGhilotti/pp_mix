#include "conditional_mcmc.hpp"


ConditionalMCMC::ConditionalMCMC(BasePP *pp_mix, BaseJump *h, BasePrec *g):
    pp_mix(pp_mix), h(h), g(g) {}

void ConditionalMCMC::initialize(const MatrixXd &data) {
    this->data = data;
    dim = data.cols();
    ndata = data.rows();

    MatrixXd ranges = pp_mix->get_ranges();

    // TODO @Mario this works only in R^2!
    a_means = MatrixXd(4, 2);
    a_means.row(0) = ranges.row(0);
    a_means.row(1) = ranges.row(1);
    a_means.row(2) = ranges.col(0).transpose();
    a_means(3, 0) = ranges(1, 1);
    a_means(3, 1) = ranges(0, 1);

    std::cout << "a_means: \n" << a_means << std::endl;

    nclus = a_means.rows();

    clus_alloc = VectorXi::Zero(ndata);
    VectorXd probas = VectorXd::Ones(nclus) / nclus;
    for (int i=0; i < ndata; i++) {
        clus_alloc(i) = categorical_rng(probas, Rng::Instance().get()) - 1;
    }

    a_jumps = VectorXd::Ones(nclus) / (nclus + 1);
    na_jumps = VectorXd::Ones(nclus) / (nclus + 1);

    a_means = pp_mix->sample_uniform(nclus);
    na_means = pp_mix->sample_uniform(nclus);

    a_precs.resize(nclus);
    na_precs.resize(nclus);

    for (int i=0; i < nclus; i++) {
        a_precs[i] = g->sample_prior();
        na_precs[i] = g->sample_prior();
    }
}

void ConditionalMCMC::run_one() {
    double T = a_jumps.sum() + na_jumps.sum();
    u = gamma_rng(ndata, T, Rng::Instance().get());
    double psi_u = h->laplace(u);

    sample_allocations_and_relabel();

    // UNALLOCATED PROCESS
    for (int i=0; i < 10; i++)
        pp_mix->sample_given_active(a_means, &na_means, psi_u);

    na_precs.resize(na_means.rows());
    na_jumps.conservativeResize(na_means.rows(), 1);
    for (int i=0; i < na_means.rows(); i++) {
        na_precs[i] = g->sample_prior();
        na_jumps(i) = h->sample_tilted(u);
    }

    // ALLOCATED PROCES
    sample_means();
    sample_vars();
    sample_jumps();


    pp_mix->update_hypers(a_means, na_means);
}

void ConditionalMCMC::sample_allocations_and_relabel() {
    int Ma = a_means.rows();
    int Mna = na_means.rows();
    int Mtot = Ma + Mna;

    const MatrixXd& curr_a_means = a_means;
    const MatrixXd& curr_na_means = na_means;
    const std::vector<PrecMat>& curr_a_precs = a_precs;
    const std::vector<PrecMat> &curr_na_precs = na_precs;
    const VectorXd& curr_a_jumps = a_jumps;
    const VectorXd &curr_na_jumps = na_jumps;

    #pragma omp parallel for
    for (int i=0; i < ndata; i++) {
        VectorXd probas(Mtot);
        // VectorXd mean;
        int newalloc;
        const VectorXd& datum = data.row(i).transpose();
        probas.head(Ma) = curr_a_jumps;
        probas.tail(Mna) = curr_na_jumps;
        probas = log(probas);

        for (int k=0; k < Ma; k++) {
            probas[k] += o_multi_normal_prec_lpdf(
                datum, curr_a_means.row(k).transpose(), curr_a_precs[k]);
        }
        for (int k = 0; k < Mna; k++) {
            probas[k + Ma] += o_multi_normal_prec_lpdf(
                datum, curr_na_means.row(k).transpose(), curr_na_precs[k]);
        }
        probas = exp(probas);
        probas /= probas.sum();
        newalloc = categorical_rng(probas, Rng::Instance().get()) - 1;
        clus_alloc[i] = newalloc;
    }

    _relabel();
}

void ConditionalMCMC::_relabel() {
    std::set<int> na2a; // non active that become active
    std::set<int> a2na; // active that become non active

    int Ma = a_means.rows();
    int Mna = na_means.rows();
    int Mtot = Ma + Mna;

    for (int i = 0; i < ndata; i++)
    {
        if (clus_alloc(i) >= Ma)
            na2a.insert(clus_alloc(i) - Ma);
    }

    // NOW WE RELABEL
    // FIND OUT WHICH CLUSTER HAVE BECOME NON-ACTIVE
    for (int k = 0; k < Ma; k++)
    {
        if ((clus_alloc.array() == k).count() == 0)
            a2na.insert(k);
    }
    std::vector<int> a2na_vec(a2na.begin(), a2na.end());
    int n_new_na = a2na.size();
    MatrixXd new_na_means(n_new_na, dim);
    std::vector<PrecMat> new_na_precs(n_new_na);
    VectorXd new_na_jumps(n_new_na);

    for (int i = 0; i < n_new_na; i++)
    {
        new_na_means.row(i) = a_means.row(a2na_vec[i]);
        new_na_precs[i] = a_precs[a2na_vec[i]];
        new_na_jumps(i) = a_jumps(a2na_vec[i]);
    }

    // NOW TAKE CARE OF NON ACTIVE THAT BECOME ACTIVE
    std::vector<int> na2a_vec(na2a.begin(), na2a.end());
    int n_new_a = na2a_vec.size();
    MatrixXd new_a_means(n_new_a, dim);
    std::vector<PrecMat> new_a_precs(n_new_a);
    VectorXd new_a_jumps(n_new_a);

    for (int i = 0; i < n_new_a; i++)
    {
        new_a_means.row(i) = na_means.row(na2a_vec[i]);
        new_a_precs[i] = na_precs[na2a_vec[i]];
        double tmp = na_jumps(na2a_vec[i]);
        new_a_jumps(i) = tmp;
    }

    // delete rows, backward
    for (auto it = a2na_vec.rbegin(); it != a2na_vec.rend(); it++)
    {
        delete_row(&a_means, *it);
        delete_elem(&a_jumps, *it);
        a_precs.erase(a_precs.begin() + *it);
    }

    // delete rows, backward
    if (na_means.rows() > 0)
    {
        for (auto it = na2a_vec.rbegin(); it != na2a_vec.rend(); it++)
        {
            delete_row(&na_means, *it);
            delete_elem(&na_jumps, *it);
            na_precs.erase(na_precs.begin() + *it);
        }
    }

    // NOW JOIN THE STUFF TOGETHER
    if (new_a_means.rows() > 0)
    {
        int oldMa = a_means.rows();
        a_means.conservativeResize(oldMa + new_a_means.rows(), dim);
        a_means.block(oldMa, 0, new_a_means.rows(), dim) = new_a_means;

        a_jumps.conservativeResize(oldMa + new_a_means.rows());
        a_jumps.segment(oldMa, new_a_means.rows()) = new_a_jumps;

        for (const auto &prec : new_a_precs)
            a_precs.push_back(prec);
    }

    if (new_na_means.rows() > 0)
    {
        int oldMna = na_means.rows();
        na_means.conservativeResize(oldMna + new_na_means.rows(), dim);
        na_means.block(oldMna, 0, new_na_means.rows(), dim) = new_na_means;

        na_jumps.conservativeResize(oldMna + new_na_means.rows());
        na_jumps.segment(oldMna, new_na_means.rows()) = new_na_jumps;
    }

    for (const auto &prec : new_na_precs)
        na_precs.push_back(prec);

    // NOW RELABEL
    std::set<int> uniques_(clus_alloc.data(), clus_alloc.data() + ndata);
    std::vector<int> uniques(uniques_.begin(), uniques_.end());
    std::map<int, int> old2new;
    for (int i = 0; i < uniques.size(); i++)
        old2new.insert({uniques[i], i});

    for (int i = 0; i < ndata; i++)
    {
        clus_alloc(i) = old2new[clus_alloc(i)];
    }

    data_by_clus.resize(0);
    data_by_clus.resize(a_means.rows());
    for (int i = 0; i < ndata; i++)
    {
        data_by_clus[clus_alloc(i)].push_back(data.row(i).transpose());
    }
}

void ConditionalMCMC::sample_means() {
    // We update each mean separately, using the Papangelou as full conditional
    for (int i=0; i < a_means.rows(); i++) {
        MatrixXd others;
        const MatrixXd& proposal_var_chol = MatrixXd::Identity(dim, dim);
        double currlik, proplik, prior_ratio, lik_ratio, arate;

        const VectorXd& currmean = a_means.row(i).transpose();
        currlik = o_multi_normal_prec_lpdf(data_by_clus[i], currmean, a_precs[i]);
        

        const VectorXd &prop = multi_normal_cholesky_rng(
            currmean, proposal_var_chol, Rng::Instance().get());
        proplik = o_multi_normal_prec_lpdf(data_by_clus[i], prop, a_precs[i]);

        

        lik_ratio = proplik - currlik;
        if (i==0) {
            prior_ratio = pp_mix->dens(prop) - pp_mix->dens(currmean);
        } else {
            others = a_means.block(0, 0, i, dim);
            prior_ratio = pp_mix->papangelou(prop, others) -
                        pp_mix->papangelou(currmean, others);
        }

        if (verbose) {
            std::cout << "Component: " << i <<  std::endl;
            std::cout << "currmean: " << currmean.transpose()
                      << ", currlik: " << currlik << std::endl;
            std::cout << "prop: " << prop.transpose()
                      << ", proplik: " << proplik << std::endl;
            std::cout << "prior_ratio: " << prior_ratio << std::endl;
            if (i > 0) {
                std::cout << "prop_papangelou: " << pp_mix->papangelou(prop, others)
                          << ", curr_papangelou: " << pp_mix->papangelou(currmean, others)
                          << std::endl;
            }
            std::cout << "lik_ratio: " << lik_ratio << std::endl;
            std::cout << "**********" << std::endl;
        }

        arate = lik_ratio + prior_ratio;
        if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < arate)
            a_means.row(i) = prop.transpose();
    }
}

void ConditionalMCMC::sample_vars()
{
    #pragma omp parallel for
    for (int i=0; i < a_means.rows(); i++)
        a_precs[i] = g->sample_given_data(
            data_by_clus[i], a_precs[i], a_means.row(i).transpose());
}

void ConditionalMCMC::sample_jumps()
{
    #pragma omp parallel for
    for (int i = 0; i < a_means.rows(); i++)
        a_jumps(i) = h->sample_given_data(data_by_clus[i], a_jumps(i), u);
}

MixtureState ConditionalMCMC::get_state_as_proto() {
    MixtureState out;
    out.set_ma(a_means.rows());
    out.set_mna(na_means.rows());
    out.set_mtot(a_means.rows() + na_means.rows());
    

    for (int i = 0; i < a_means.rows(); i++)
    {
        EigenVector *mean;
        EigenMatrix *prec;
        mean = out.add_a_means();
        prec = out.add_a_precs();

        to_proto(a_means.row(i).transpose(), mean);
        to_proto(a_precs[i].get_prec(), prec);
    }

    for (int i=0; i < na_means.rows(); i++) {
        EigenVector *mean;
        EigenMatrix *prec;
        mean = out.add_na_means();
        prec = out.add_na_precs();

        to_proto(na_means.row(i).transpose(), mean);
        to_proto(na_precs[i].get_prec(), prec);
    }

    to_proto(a_jumps, out.mutable_a_jumps());
    to_proto(na_jumps, out.mutable_na_jumps());

    *out.mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() + ndata};

    StraussState pp_params;
    pp_mix->get_state_as_proto(&pp_params);
    out.mutable_pp_state()->CopyFrom(pp_params);
    return out;
}

void ConditionalMCMC::print_debug_string() {
    std::cout << "#### ACTIVE: Number actives: " << a_means.rows();
    for (int i=0; i < a_means.rows(); i++) {
        std::cout << "Component: " << i 
                  << ", weight: " << a_jumps(i)
                  << ", mean: " << a_means.row(i)
                  << std::endl;
        // std::cout << "Data: ";
        // for (int j=0; j < data_by_clus[i].size(); j++)
        //     std::cout << "[" << data_by_clus[i][j].transpose() << "],    ";

        std::cout << std::endl;
    }

    std::cout << "#### NON - ACTIVE: Number actives: " << na_means.rows();
    for (int i = 0; i < na_means.rows(); i++)
    {
        std::cout << "Component: " << i
                  << "weight: " << na_jumps(i)
                  << ", mean: " << na_means.row(i)
                  << std::endl;
    }

    std::cout << std::endl;
}