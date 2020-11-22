#ifndef RJ_MCMC
#define RJ_MCMC

#include <google/protobuf/message.h>
#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <stan/math/prim.hpp>
#include <vector>

#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"
#include "jumps/base_jump.hpp"
#include "point_process/base_pp.hpp"
#include "precs/base_prec.hpp"
#include "precs/fake_prec.hpp"
#include "precs/precmat.hpp"
#include "rng.hpp"
#include "utils.hpp"

using namespace Eigen;

template <class Prec, typename prec_t, typename data_t>
class RJMCMC {
 protected:
  int dim;
  int ndata;
  std::vector<data_t> data;
  double prior_ratio, lik_ratio;
  std::vector<std::vector<data_t>> data_by_clus;

  // STATE
  int nclus;
  VectorXi clus_alloc;
  VectorXd weights;
  MatrixXd means;
  std::vector<prec_t> precs;

  // DISTRIBUTIONS
  BasePP *pp_mix;
  Prec *g;

  // FOR DEBUGGING
  bool verbose = false;
  int acc_mean = 0;
  int tot_mean = 0;

  Params params;
  double min_proposal_sigma, max_proposal_sigma;
  double prior_dir = 1.0;

 public:
  RJMCMC() {}

  ~RJMCMC() {
    delete pp_mix;
    delete g;
  }

  RJMCMC(BasePP *pp_mix, Prec *g, const Params &params);

  void set_pp_mix(BasePP *pp_mix) { this->pp_mix = pp_mix; }
  void set_jump(BaseJump *h) { this->h = h; }
  void set_prec(Prec *g) { this->g = g; }

  void initialize(const std::vector<data_t> &data);

  virtual void initialize_allocated_means() = 0;

  void run_one();

  void sample_allocations();

  void sample_means();

  void sample_vars();

  void sample_weights();

  virtual void combine() = 0;

  virtual void split() = 0;

  virtual void get_state_as_proto(google::protobuf::Message *out_) = 0;

  void print_debug_string();

  void set_verbose() { verbose = !verbose; }

  virtual double lpdf_given_clus(const data_t &x, const VectorXd &mu,
                                 const prec_t &sigma) = 0;

  virtual double lpdf_given_clus_multi(const std::vector<data_t> &x,
                                       const VectorXd &mu,
                                       const prec_t &sigma) = 0;

  virtual void set_dim(const data_t &datum) = 0;

  virtual VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) = 0;

  double mean_acceptance_rate() { return (1.0 * acc_mean) / (1.0 * tot_mean); }

  virtual void print_data_by_clus(int clus) = 0;

};

class UnivariateRJMCMC : public RJMCMC<BaseUnivPrec, double, double> {
 public:
  UnivariateRJMCMC() {}

  UnivariateRJMCMC(BasePP *pp_mix, BasePrec *g,
                   const Params &params);

  void initialize_allocated_means() override;

  void get_state_as_proto(google::protobuf::Message *out_) override;

  double lpdf_given_clus(const double &x, const VectorXd &mu,
                         const double &sigma) {
    return stan::math::normal_lpdf(x, mu(0), 1.0 / sqrt(sigma));
  }

  double lpdf_given_clus(const double &x, const double &mu,
                         const double &sigma) {
    return stan::math::normal_lpdf(x, mu, 1.0 / sqrt(sigma));
  }

  double lpdf_given_clus_multi(const std::vector<double> &x, const VectorXd &mu,
                               const double &sigma) {
    return stan::math::normal_lpdf(x, mu(0), 1.0 / sqrt(sigma));
  }

  void set_dim(const double &datum) {
    std::cout << "set_dim" << std::endl;
    dim = 1;
    std::cout << dim << std::endl;
  }

  VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) override;

  void print_data_by_clus(int clus);

  void combine() override;

  void split() override;
};

// class MultivariateRJMCMC : public RJMCMC<BaseMultiPrec, PrecMat, VectorXd> {
//  public:
//   MultivariateRJMCMC() {}

//   MultivariateRJMCMC(BasePP *pp_mix, BasePrec *g, const Params &params);

//   void initialize_allocated_means() override;

//   void get_state_as_proto(google::protobuf::Message *out_) override;

//   double lpdf_given_clus(const VectorXd &x, const VectorXd &mu,
//                          const PrecMat &sigma) {
//     return o_multi_normal_prec_lpdf(x, mu, sigma);
//   }

//   double lpdf_given_clus_multi(const std::vector<VectorXd> &x,
//                                const VectorXd &mu, const PrecMat &sigma) {
//     return o_multi_normal_prec_lpdf(x, mu, sigma);
//   }

//   VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) override;

//   void set_dim(const VectorXd &datum) { dim = datum.size(); }

//   void print_data_by_clus(int clus);

//   void combine() override;

//   void split() override;
// };

#include "rj_mcmc_imp.hpp"

#endif
