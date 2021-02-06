#ifndef ALLOC_MEANS_SAMPLER
#define ALLOC_MEANS_SAMPLER

#include "rng.hpp"
#include "conditional_mcmc.hpp"

using namespace Eigen;
using namespace stan::math;

namespace MCMCsampler {

class BaseMeansSampler {
protected:
    MultivariateConditionalMCMC* mcmc;

    int acc_sampled_a_means = 0;
    int tot_sampled_a_means = 0;
   

public:
    BaseMeansSampler(MultivariateConditionalMCMC* mcmc): mcmc(mcmc) {}
    virtual ~BaseMeansSampler(){}

    virtual void operator() = 0;

    double Means_acc_rate();
};


class MeansSamplerClassic : public BaseMeansSampler {
private:
    double prop_means_sigma;

public:
    MeansSamplerClassic(MultivariateConditionalMCMC* mcmc, double p_m_s): BaseMeansSampler(mcmc), prop_means_sigma(p_m_s){}
    void operator() override;
};

class MeansSamplerMala : public BaseMeansSampler {
private:
    double mala_p_means;

public:
    LambdaSamplerMala(MultivariateConditionalMCMC* mcmc, double m_p): BaseLambdaSampler(mcmc), mala_p_means(m_p){}
    void operator() override;
    
    // TARGET FUNCTION OBJECT : must implement logfunction (as required in Mala)
    class means_target_function {
    private:
        const MultivariateConditionalMCMC& m_mcmc;
       
    public:
   
        means_target_function(const MultivariateConditionalMCMC& mala): m_mcmc(mala){};

        template<typename T> T
        operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & lamb) const ;
    } means_tar_fun;
};

};

#include "alloc_means_sampler_imp.hpp"

#endif