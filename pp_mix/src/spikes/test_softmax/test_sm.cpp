//#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
//#include <stan/math/prim.hpp>
#include <stan/math/fwd.hpp>
#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>

#include "../../conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../precs/delta_gamma.hpp"
#include "../../precs/delta_wishart.hpp"
#include "../../point_process/determinantalPP.hpp"

using namespace Eigen;
using namespace stan::math;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

int main() {

    MatrixXd datacsv = load_csv<MatrixXd>("/home/lorenzo/Documents/Tesi/github_repos/pp_mix/data/Student_data/datasets/stud_p_400_d_8_M_4_npc_50_data.csv");

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // NOTE: We use all params
    int d = 8;

    DeterminantalPP* pp_mix = make_dpp(params, d);
    BasePrec* g = make_delta(params, d);

    MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params, d);

    sampler.initialize(datacsv);

    int log_every=10;
    int ntrick = 100;
    int burnin = 100;
    int niter=100;
    int thin = 1;

    for (int i = 0; i < ntrick; i++) {
        sampler.run_one_trick();
        if ((i + 1) % log_every == 0) {
            std::cout<< "Trick, iter #"<< i + 1<< " / "<< ntrick<<"\n";
            //myfile<< "Lambda: \n"<< sampler.get_Lambda()<<"\n";
            //myfile<< "Grad_log_ad: \n"<< sampler.get_grad_log_ad()<<"\n";
            //myfile<< "Grad_log_analytic: \n"<< sampler.get_grad_log_analytic()<<"\n";

        }
    }

    for (int i = 0; i < burnin; i++) {
        sampler.run_one();
        if ((i + 1) % log_every == 0) {
            std::cout<<"Burnin, iter #"<< i + 1<< " / "<< burnin<<std::endl;
            //myfile<< "Means_na: \n"<< sampler.get_na_means()<<"\n";
            //myfile<< "Lambda: \n"<< sampler.get_Lambda()<<"\n";
            /*myfile<< "diff_log_dens_ad: \n"<< sampler.get_ln_dens_ad()<<"\n";
            myfile<< "diff_log_dens_analytic: \n"<< sampler.get_ln_dens_analytic()<<"\n";

            myfile<< "Grad_log_ad: \n"<< sampler.get_grad_log_ad()<<"\n";
            myfile<< "Grad_log_analytic: \n"<< sampler.get_grad_log_analytic()<<"\n";
          */}
    }

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        if ((i + 1) % log_every == 0) {
            std::cout<<"Running, iter #"<< i + 1<< " / "<< niter<<std::endl;
            //myfile<< "Means_na: \n"<< sampler.get_na_means()<<"\n";
            //myfile<< "Lambda: \n"<< sampler.get_Lambda()<<"\n";
          /*  myfile<< "diff_log_dens_ad: \n"<< sampler.get_ln_dens_ad()<<"\n";
            myfile<< "diff_log_dens_analytic: \n"<< sampler.get_ln_dens_analytic()<<"\n";

            myfile<< "Grad_log_ad: \n"<< sampler.get_grad_log_ad()<<"\n";
            myfile<< "Grad_log_analytic: \n"<< sampler.get_grad_log_analytic()<<"\n";
      */    }
    }

    //myfile.close();
    std::cout<<"acceptance lambda: "<<sampler.Lambda_acceptance_rate()<<std::endl;


    std::cout<<"END!"<<std::endl;
    return 0;


}
