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
std::cout<<"begin"<<std::endl;
    MatrixXd datacsv = load_csv<MatrixXd>("/home/beraha/pp_mix/data/Student_data/datasets/datasets/stud_p_200_d_2_M_4_npc_50_data.csv");
    std::cout<<"here2"<<std::endl;
    std::string params_file = \
      "/home/beraha/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // NOTE: We use all params
    std::cout<<"here3"<<std::endl;
    int d = 8;

    DeterminantalPP* pp_mix = make_dpp(params, d);
    std::cout<<"here4"<<std::endl;
    BasePrec* g = make_delta(params, d);
std::cout<<"here5"<<std::endl;
    MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params, d);
std::cout<<"here6"<<std::endl;
    sampler.initialize(datacsv);

    Eigen::VectorXi init_allocs_ = VectorXi::Zero(datacsv.rows());
    sampler.set_clus_alloc(init_allocs_);
    sampler._relabel();

std::cout<<"here7"<<std::endl;
    int log_every=10;
    int ntrick = 0;
    int burnin = 0;
    int niter=50;
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
