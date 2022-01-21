#ifndef LAMBDA_SAMPLER_HPP
#define LAMBDA_SAMPLER_HPP

#include <omp.h>
#include <stan/math/fwd.hpp>
#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>
#include "rng.hpp"

using namespace Eigen;
using namespace stan::math;
/*
namespace MCMCsampler {
    class MultivariateConditionalMCMC ;
}*/
#include "conditional_mcmc.hpp"

namespace MCMCsampler {

class BaseLambdaSampler {
protected:
    MultivariateConditionalMCMC* mcmc;

    int acc_sampled_Lambda = 0;
    int tot_sampled_Lambda = 0;
    double norm_d_grad=0;

public:
    BaseLambdaSampler(MultivariateConditionalMCMC* mcmc): mcmc(mcmc) {}
    virtual ~BaseLambdaSampler(){}

    virtual void perform() = 0;
    double get_norm_d_g(){return norm_d_grad;}
    double Lambda_acc_rate();
};


class LambdaSamplerClassic : public BaseLambdaSampler {
private:
    double prop_lambda_sigma;

     // for update of Lambda!
    inline double compute_exp_lik(const MatrixXd& lamb) const;
    inline double compute_exp_prior(const MatrixXd& lamb) const;

public:
    LambdaSamplerClassic(MultivariateConditionalMCMC* mcmc, double p_l_s): BaseLambdaSampler(mcmc), prop_lambda_sigma(p_l_s){}
    void perform() override;
};

class LambdaSamplerMala : public BaseLambdaSampler {
private:
    double mala_p_lambda;

    MatrixXd compute_grad_analytic(){};

public:
    LambdaSamplerMala(MultivariateConditionalMCMC* mcmc, double m_p): BaseLambdaSampler(mcmc), mala_p_lambda(m_p), lambda_tar_fun(*mcmc){}
    void perform() override;

    // TARGET FUNCTION OBJECT : must implement logfunction (as required in Mala)
    class lambda_target_function {
    private:
        const MultivariateConditionalMCMC& m_mcmc;

    public:

        lambda_target_function(const MultivariateConditionalMCMC& mala): m_mcmc(mala){};

        template<typename T> T
        operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & lamb) const ;
    } lambda_tar_fun;
};

}

#include "lambda_sampler_imp.hpp"

#endif
