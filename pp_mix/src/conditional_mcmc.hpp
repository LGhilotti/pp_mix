#ifndef CONDITIONAL_MCMC
#define CONDITIONAL_MCMC

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <functional>

#include <Eigen/Dense>
#include <stan/math/prim/mat.hpp>
#include <google/protobuf/message.h>

#include "rng.hpp"
#include "point_process/base_pp.hpp"
#include "jumps/base_jump.hpp"
#include "precs/base_prec.hpp"
#include "precs/fake_prec.hpp"
#include "precs/precmat.hpp"
#include "utils.hpp"
#include "../protos/cpp/state.pb.h"
#include "../protos/cpp/params.pb.h"


using namespace Eigen;


template<class Prec, typename prec_t, typename data_t>
class ConditionalMCMC {
 protected:
     int dim;
     int ndata;
     std::vector<data_t> data;
     double prior_ratio, lik_ratio;
     std::vector<std::vector<data_t>> data_by_clus;

     // STATE
     int nclus;
     double u;
     VectorXi clus_alloc;
     VectorXd a_jumps, na_jumps;
     MatrixXd a_means, na_means;
     std::vector<prec_t> a_precs, na_precs;

     // DISTRIBUTIONS
     BasePP *pp_mix;
     BaseJump *h;
     Prec *g;

     // FOR DEBUGGING
     bool verbose = false;
     int acc_mean = 0;
     int tot_mean = 0;

     Params params;

     double min_proposal_sigma, max_proposal_sigma;

 public:
     ConditionalMCMC() {}
     ~ConditionalMCMC()
     {
         delete pp_mix;
         delete h;
         delete g;
    }

    ConditionalMCMC(
        BasePP * pp_mix, BaseJump * h, Prec * g, 
        const Params& params);

    void set_pp_mix(BasePP* pp_mix) {this->pp_mix = pp_mix;}
    void set_jump(BaseJump* h) {this->h = h;}
    void set_prec(Prec* g) {this->g = g;}

    void initialize(const std::vector<data_t> &data);

    virtual void initialize_allocated_means() = 0;

    void run_one();

    void sample_allocations_and_relabel();

    void sample_means();

    void sample_vars();

    void sample_jumps();

    virtual void get_state_as_proto(google::protobuf::Message *out_) = 0;

    void print_debug_string();


    void _relabel();

    void set_verbose() { verbose = !verbose; }

    virtual double lpdf_given_clus(
        const data_t &x, const VectorXd &mu, const prec_t &sigma)  = 0;

    virtual double lpdf_given_clus_multi(
        const std::vector<data_t> &x, const VectorXd &mu, 
        const prec_t &sigma) = 0;

    virtual void set_dim(const data_t& datum) = 0;

    virtual VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) = 0;

    double mean_acceptance_rate() {
        return (1.0 * acc_mean) / (1.0 * tot_mean);
    }

    virtual void print_data_by_clus(int clus) = 0;
};


class MultivariateConditionalMCMC: public ConditionalMCMC<
    BaseMultiPrec, PrecMat, VectorXd> {

public:
    MultivariateConditionalMCMC() {}

    MultivariateConditionalMCMC(BasePP *pp_mix, BaseJump *h, BasePrec *g,
                                const Params &params);

    void initialize_allocated_means() override;

    void get_state_as_proto(google::protobuf::Message *out_) override;

    double lpdf_given_clus(
        const VectorXd &x, const VectorXd &mu, const PrecMat &sigma)
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }

    double lpdf_given_clus_multi(
        const std::vector<VectorXd> &x, const VectorXd &mu, const PrecMat &sigma)
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }

    VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) override;

    void set_dim(const VectorXd& datum) {
        dim = datum.size();
    }

    void print_data_by_clus(int clus); 

};

class UnivariateConditionalMCMC : public ConditionalMCMC<
    BaseUnivPrec, double, double>
{
public:
    UnivariateConditionalMCMC() {}

    UnivariateConditionalMCMC(BasePP *pp_mix, BaseJump *h, BasePrec *g,
                              const Params &params);

    void initialize_allocated_means() override;

    void get_state_as_proto(google::protobuf::Message *out_) override;

    double lpdf_given_clus(
        const double &x, const VectorXd &mu, const double &sigma)
    {
        return stan::math::normal_lpdf(x, mu(0), 1.0 / sqrt(sigma));
    }

    double lpdf_given_clus_multi(
        const std::vector<double> &x, const VectorXd &mu, const double &sigma)
    {
        return stan::math::normal_lpdf(x, mu(0), 1.0 / sqrt(sigma));
    }

    void set_dim(const double &datum)
    {
        std::cout << "set_dim" << std::endl;
        dim = 1;
        std::cout << dim << std::endl;
    }

    VectorXd compute_grad_for_clus(int clus, const VectorXd& mean) override;

    void print_data_by_clus(int clus);
};

class BernoulliConditionalMCMC
    : public ConditionalMCMC<FakePrec, PrecMat, VectorXd> {
 public:
  BernoulliConditionalMCMC() {}

  BernoulliConditionalMCMC(BasePP *pp_mix, BaseJump *h, BasePrec *g,
                           const Params &params);

  void initialize_allocated_means() override;

  void get_state_as_proto(google::protobuf::Message *out_) override;

  double lpdf_given_clus(const VectorXd &x, const VectorXd &mu,
                         const PrecMat &sigma);

  double lpdf_given_clus_multi(const std::vector<VectorXd> &x,
                               const VectorXd &mu, const PrecMat &sigma);

  VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) override;

  void set_dim(const VectorXd &datum) { dim = datum.size(); }

  void print_data_by_clus(int clus);
};

#include "conditional_mcmc_imp.hpp"

#endif