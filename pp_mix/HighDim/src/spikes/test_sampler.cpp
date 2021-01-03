#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>

#include <stan/math/prim.hpp>

#include "../conditional_mcmc.hpp"
#include "../factory.hpp"
#include "../precs/delta_gamma.hpp"
#include "../precs/delta_wishart.hpp"
#include "../../protos/cpp/state.pb.h"
#include "../../protos/cpp/params.pb.h"
#include "../point_process/multi_factor_dpp.hpp"
#include "../point_process/uni_factor_dpp.hpp"


Eigen::MatrixXd simulate_multivariate() {
    int dim = 2;
    int data_per_clus = 10;
    Eigen::MatrixXd data = Eigen::MatrixXd(data_per_clus * 2, dim);
    Eigen::VectorXd mean1 = Eigen::VectorXd::Ones(dim) * 5.0;

    Eigen::VectorXd mean2 = Eigen::VectorXd::Ones(dim) * (-5.0);

    double sigma = 0.3;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(dim, dim) * sigma;


    for (int i = 0; i < data_per_clus; i++)
    {
        data.row(i) = stan::math::multi_normal_rng(
            mean1, cov, Rng::Instance().get());
    }

    for (int i = data_per_clus; i < 2 * data_per_clus; i++)
    {
        data.row(i) = stan::math::multi_normal_rng(
            mean2, cov, Rng::Instance().get());
    }

    return data;
}


Eigen::MatrixXd simulate_univariate() {
    int data_per_clus = 50;
    Eigen::MatrixXd data = Eigen::MatrixXd(data_per_clus * 2, 1);

    for (int i = 0; i < data_per_clus; i++)
    {
        data(i) = stan::math::normal_rng(
            5, 0.8, Rng::Instance().get());
    }

    for (int i = data_per_clus; i < 2 * data_per_clus; i++)
    {
        data(i) = stan::math::normal_rng(
            -5, 0.8, Rng::Instance().get());
    }

    return data;
}


int main() {
    //Eigen::MatrixXd data = simulate_univariate();
    Eigen::MatrixXd data = simulate_multivariate();

    Eigen::MatrixXd ranges(2, data.cols());
    ranges.row(0) = data.colwise().minCoeff();
    ranges.row(1) = data.colwise().maxCoeff();
    ranges *= 2;

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/HighDim/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);

    BaseDeterminantalPP *pp_mix = make_dpp(params, ranges);


    BasePrec *g = make_delta(params);

    //UnivariateConditionalMCMC sampler(pp_mix, g, params);
    MultivariateConditionalMCMC sampler(pp_mix, g, params);

    sampler.initialize(data);

    std::cout<<"ALL GOOD!"<<std::endl;
    
/*
    // std::deque<MultivariateMixtureState> chains;

    // sampler.set_verbose();
    // for (int i = 0; i < 100000; i++) {
    //     sampler.run_one();
    // }
    // sampler.print_debug_string();
    // sampler.set_verbose();
    int niter = 100000;
    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        // MultivariateMixtureState state;
        // sampler.get_state_as_proto(&state);
        // chains.push_back(state);
        // if (i % 1000 == 0)
        //     sampler.print_debug_string();
        // sampler.print_debug_string();

        if (i % 100 == 0) {
            std::cout << "iter: " << i << " / " << niter << std::endl;
        }
    }
    sampler.print_debug_string();
    std::cout << "ACCEPTANCE RATE: " << sampler.mean_acceptance_rate() << std::endl;
*/
}
