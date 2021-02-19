#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
//#include <stan/math/prim.hpp>
//#include <stan/math/fwd.hpp>
//#include <stan/math/mix.hpp>
//#include <stan/math/prim.hpp>

#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"
#include "../../conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../utils.hpp"
//#include "../../precs/delta_gamma.hpp"
//#include "../../precs/delta_wishart.hpp"
//#include "../../point_process/determinantalPP.hpp"
#include <Eigen/Dense>

using namespace Eigen;
using namespace stan::math;


MatrixXd generate_etas(const MatrixXd& mus, const std::vector<MatrixXd>& deltas,
                        const VectorXi& c_alloc) {

    MatrixXd out (c_alloc.size(),mus.cols());

    for (int i = 0; i < c_alloc.size(); i++) {
      out.row(i) = stan::math::multi_normal_prec_rng(mus.row(c_alloc(i)), deltas[c_alloc(i)], Rng::Instance().get());
    }

    return out;
}


MatrixXd generate_data(const MatrixXd& lambda, const MatrixXd& etas, const VectorXd& sigma_bar) {

  MatrixXd out(etas.rows(),lambda.rows());
  MatrixXd means = lambda*etas.transpose();
  MatrixXd sigma_bar_mat = sigma_bar.asDiagonal();
  for (int i = 0; i < etas.rows(); i ++ ) {
    out.row(i) = stan::math::multi_normal_prec_rng(means.col(i),sigma_bar_mat, Rng::Instance().get());
  }

  return out;

}


int main() {

    // we consider p=4, d=2
    // we fix Lambda, M=3 (3 groups), mu1,mu2,mu3 Delta1, Delta2,Delta3 cluster labels, Sigmabar
    // and simulate etas and data.
    const int p = 50;
    const int d = 2;
    MatrixXd Lambda = MatrixXd::Zero(p,d);
    Lambda.block(0,0,25,1) = VectorXd::Ones(25);
    Lambda.block(25,1,25,1) = - VectorXd::Ones(25);

    const int M = 3;
    VectorXd mu_0(d), mu_1(d), mu_2(d);
    mu_0 << -5. , 10. ;
    mu_1 << 0. , -5. ;
    mu_2 << 5. , -5. ;
    MatrixXd Mus(M,d);
    Mus.row(0)=mu_0;
    Mus.row(1)=mu_1;
    Mus.row(2)=mu_2;

    MatrixXd Delta_0(d,d), Delta_1(d,d), Delta_2(d,d);
    Delta_0 = MatrixXd::Identity(d,d);
    Delta_1 = MatrixXd::Identity(d,d);
    Delta_2 = MatrixXd::Identity(d,d);
    std::vector<MatrixXd> Deltas{Delta_0,Delta_1,Delta_2};

    VectorXd sigma_bar (VectorXd::Constant(p, 2.0));

    // we will consider 120 observations
    VectorXi cluster_alloc(120);
    cluster_alloc.head(40) = VectorXi::Zero(40);
    cluster_alloc.segment(40,40) = VectorXi::Ones(40);
    cluster_alloc.tail(40) = VectorXi::Constant(40, 2);

      MatrixXd Etas = generate_etas(Mus, Deltas, cluster_alloc);

      MatrixXd data = generate_data(Lambda, Etas, sigma_bar);


      Eigen::MatrixXd ranges(2, d);
      ranges.row(0) = RowVectorXd::Constant(d, -50);
      ranges.row(1) = RowVectorXd::Constant(d, 50);

    // UNTILL HERE, JUST GENERATION OF DATA AND LATENTS

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // NOTE: We use all params


    int log_every=10;
    int ntrick = 10;
    int burnin = 10;
    int niter=10;
    int thin = 1;


    DeterminantalPP* pp_mix = make_dpp(params, ranges);

    BasePrec* g = make_delta(params);

    MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params);

    sampler.initialize(data);

    for (int i = 0; i < ntrick; i++) {
        sampler.run_one_trick();
        if ((i + 1) % log_every == 0) {
            std::cout<< "Trick, iter #"<< i + 1<< " / "<< ntrick<<std::endl;
        }
    }

    for (int i = 0; i < burnin; i++) {
        sampler.run_one();
        if ((i + 1) % log_every == 0) {
            std::cout<<"Burnin, iter #"<< i + 1<< " / "<< burnin<<std::endl;
        }
    }

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        if (i % thin == 0) {
            std::cout<<"taken!"<<std::endl;
        }

        if ((i + 1) % log_every == 0) {
            std::cout<<"Running, iter #"<< i + 1<< " / "<< niter<<std::endl;
        }
    }
    std::cout<<"END!"<<std::endl;
    return 0;
}
