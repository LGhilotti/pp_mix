#ifndef LAMBDA_SAMPLER
#define LAMBDA_SAMPLER

#include "rng.hpp"
#include "conditional_mcmc.hpp"

using namespace Eigen;
using namespace stan::math;

namespace MCMCsampler {

class BaseLambdaSampler {
protected:
    MultivariateConditionalMCMC* mcmc;

    int acc_sampled_Lambda = 0;
    int tot_sampled_Lambda = 0;


public:
    BaseLambdaSampler(MultivariateConditionalMCMC* mcmc): mcmc(mcmc) {}
    virtual ~BaseLambdaSampler(){}

    virtual void operator() = 0;

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
    void operator() override;
};

class LambdaSamplerMala : public BaseLambdaSampler {
private:
    double mala_p_lambda;

public:
    LambdaSamplerMala(MultivariateConditionalMCMC* mcmc, double m_p): BaseLambdaSampler(mcmc), mala_p_lambda(m_p){}
    void operator() override;
    
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

};

#include "lambda_sampler_imp.hpp"

#endif