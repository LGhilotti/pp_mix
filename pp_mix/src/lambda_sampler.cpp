#include "lambda_sampler.hpp"
#include "conditional_mcmc.hpp"

namespace MCMCsampler {
void BaseLambdaSampler::Lambda_acc_rate(){
        return (1.0 * acc_sampled_Lambda) / (1.0 * tot_sampled_Lambda);
}

//////////////////////////////////////
/// LambdaSamplerClassic /////////
///////////////////////////////

double LambdaSamplerClassic::compute_exp_lik(const MatrixXd& lamb) const {

  return (mcmc->sigma_bar.array().sqrt().matrix().asDiagonal()*(mcmc->data.transpose() - lamb*mcmc->etas.transpose())).colwise().squaredNorm().sum();

}

double LambdaSamplerClassic::compute_exp_prior(const MatrixXd& lamb) const {

  return (lamb.array().square()/(mcmc->Psi.array()*mcmc->Phi.array().square())).sum() * (- 0.5 / (mcmc->tau*mcmc->tau));

}


void LambdaSamplerClassic::perform() {
  //std::cout<<"sample Lambda"<<std::endl;
  // Current Lambda (here are the means) are expanded to vector<double> column major
  MatrixXd prop_lambda = Map<MatrixXd>(normal_rng( std::vector<double>(mcmc->get_Lambda().data(), mcmc->get_Lambda().data() + mcmc->get_dim_data()*mcmc->get_dim_fact) ,
              std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(), prop_lambda_sigma), Rng::Instance().get()).data() , mcmc->get_dim_data(), mcmc->get_dim_fact());
  // DEBUG
  //std::cout<<"Proposal Lambda: \n"<<prop_lambda<<std::endl;
//  std::cout<<"Proposed Lambda"<<std::endl;

  tot_sampled_Lambda += 1;
  // we use log for each term
  double curr_lik, prop_lik;
  curr_lik = -0.5 * compute_exp_lik(mcmc->get_Lambda());
  prop_lik = -0.5 * compute_exp_lik(prop_lambda);
  // DEBUG
  //std::cout<<"curr_lik = "<<curr_lik<<"  ; prop_lik = "<<prop_lik<<std::endl;

  double curr_prior_cond_process, prop_prior_cond_process;
  MatrixXd means(mcmc->get_num_a_means()+mcmc->get_num_na_means(),mcmc->get_dim_fact());
  means << mcmc->get_a_means(), mcmc->get_na_means();

  mcmc->pp_mix->decompose_proposal(prop_lambda);

  curr_prior_cond_process = mcmc->pp_mix->dens_cond(means, true);
  prop_prior_cond_process = mcmc->pp_mix->dens_cond_in_proposal(means, true);
  // DEBUG
  //std::cout<<"curr_p_c_p = "<<curr_prior_cond_process<<"  ; prop_p_c_p = "<<prop_prior_cond_process<<std::endl;

  double curr_prior_lambda, prop_prior_lambda;
  curr_prior_lambda = compute_exp_prior(mcmc->get_Lambda());
  prop_prior_lambda = compute_exp_prior(prop_lambda);
  // DEBUG
 //std::cout<<"curr_prior_lambda = "<<curr_prior_lambda<<" ; prop_prior_lambda = "<<prop_prior_lambda<<std::endl;

  double curr_dens, prop_dens, log_ratio;
  curr_dens = curr_lik + curr_prior_cond_process + curr_prior_lambda;
  prop_dens = prop_lik + prop_prior_cond_process + prop_prior_lambda;
  // DEBUG
//  std::cout<<"curr_dens = "<<curr_dens<<" ; prop_dens = "<<prop_dens<<std::endl;

  log_ratio = prop_dens - curr_dens;
  // DEBUG
  //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

  if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < log_ratio){
    //ACCEPTED
    acc_sampled_Lambda += 1;
    mcmc->pp_mix->decompose_proposal(prop_lambda);
    //Lambda.swap(prop_lambda);
    //mcmc->Lambda = prop_lambda;  
    mcmc->set_Lambda() = prop_lambda;
    mcmc->pp_mix->update_decomposition_from_proposal();
    //std::cout<<"accepted Lambda"<<std::endl;
  }

  return;
}


//////////////////////////////////////
/// LambdaSamplerMala /////////
///////////////////////////////


void LambdaSamplerMala::perform() {
  //std::cout<<"sample Lambda"<<std::endl;
  // Current Lambda (here are the means) are expanded to vector<double> column major
  double ln_px_curr;
  VectorXd grad_ln_px_curr;
  VectorXd Lambda_curr = Map<VectorXd>(mcmc->get_Lambda().data(), mcmc->get_dim_data()*mcmc->get_dim_fact()); // column-major
 //std::cout<<"before gradient"<<std::endl;
  stan::math::gradient(lambda_tar_fun, Lambda_curr, ln_px_curr, grad_ln_px_curr);
  //std::cout<<"after gradient"<<std::endl;
  // Proposal according MALA
  VectorXd prop_lambda_vec = Lambda_curr + mala_p_lambda*grad_ln_px_curr +
                    std::sqrt(2*mala_p_lambda)*Map<VectorXd>(normal_rng(std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(), 0.),
                  std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(),1.), Rng::Instance().get()).data() , mcmc->get_dim_data()*mcmc->get_dim_fact());


  double ln_px_prop;
  VectorXd grad_ln_px_prop;
  stan::math::gradient(lambda_tar_fun, prop_lambda_vec, ln_px_prop, grad_ln_px_prop);

  MatrixXd prop_lambda = Map<MatrixXd>(prop_lambda_vec.data(), mcmc->get_dim_data(), mcmc->get_dim_fact());

  // DEBUG
 // std::cout<<"Proposal Lambda: \n"<<prop_lambda<<std::endl;
//  std::cout<<"Proposed Lambda"<<std::endl;

  tot_sampled_Lambda += 1;
  // COMPUTE ACCEPTANCE PROBABILITY
  // (log) TARGET DENSITY TERMS
  double ln_ratio_target;
  ln_ratio_target = ln_px_prop - ln_px_curr;
  // DEBUG
  //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

  //(log) PROPOSAL DENSITY TERMS
  double ln_q_num, ln_q_den;
  ln_q_num = - (Lambda_curr - prop_lambda_vec - mala_p_lambda*grad_ln_px_prop).squaredNorm() /(4*mala_p_lambda);
  ln_q_den = - (prop_lambda_vec - Lambda_curr - mala_p_lambda*grad_ln_px_curr).squaredNorm() /(4*mala_p_lambda);

  // -> acceptance probability
  double ln_ratio = ln_ratio_target + ln_q_num - ln_q_den;

  if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < ln_ratio){
    //ACCEPTED
    acc_sampled_Lambda += 1;
    mcmc->pp_mix->decompose_proposal(prop_lambda);
    //Lambda.swap(prop_lambda);
    //mcmc->Lambda = prop_lambda;
    mcmc->set_Lambda() = prop_lambda;
    mcmc->pp_mix->update_decomposition_from_proposal();
    //std::cout<<"accepted Lambda"<<std::endl;
  }

  return;
}

}