#include "alloc_means_sampler.hpp"

namespace MCMCsampler {
void BaseMeansSampler::Means_acc_rate(){
        return (1.0 * acc_sampled_a_means) / (1.0 * tot_sampled_a_means);
}

//////////////////////////////////////
/// MeansSamplerClassic /////////
///////////////////////////////

void MeansSamplerClassic::operator() {

  MatrixXd allmeans(mcmc->a_means.rows() + mcmc->na_means.rows(), mcmc->dim_fact);
  allmeans << mcmc->a_means, mcmc->na_means;

  for (int i = 0; i < mcmc->a_means.rows(); i++) {
    tot_sampled_a_means += 1;
    MatrixXd others(allmeans.rows() - 1, mcmc->dim_fact);

    double currlik, proplik, prior_ratio, lik_ratio, arate;
    const VectorXd &currmean = mcmc->a_means.row(i).transpose();
    const MatrixXd &cov_prop = MatrixXd::Identity(mcmc->dim_fact, mcmc->dim_fact) * prop_means_sigma;

    // we PROPOSE a new point from a multivariate normal, with mean equal to the current point
    // and covariance matrix diagonal
    VectorXd prop =
        stan::math::multi_normal_rng(currmean, cov_prop, Rng::Instance().get());

    if (mcmc->is_inside(prop)){ // if not, just keep the current mean and go to the next a_mean
        currlik = mcmc->lpdf_given_clus_multi(mcmc->etas_by_clus[i], currmean, mcmc->a_deltas[i]);
        proplik = mcmc->lpdf_given_clus_multi(mcmc->etas_by_clus[i], prop, mcmc->a_deltas[i]);

        lik_ratio = proplik - currlik;
        others = delete_row(allmeans, i);

        prior_ratio =
            mcmc->pp_mix->papangelou(prop, others) - mcmc->pp_mix->papangelou(currmean, others);

        arate = lik_ratio + prior_ratio;

        if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < arate) {
          mcmc->a_means.row(i) = prop.transpose();
          acc_sampled_a_means += 1;
        }
    }

  }
  return;
}


//////////////////////////////////////
/// MeansSamplerMala /////////
///////////////////////////////


void MeansSamplerMala::operator() {
/*
  //std::cout<<"sample Lambda"<<std::endl;
  // Current Lambda (here are the means) are expanded to vector<double> column major
  double ln_px_curr;
  VectorXd grad_ln_px_curr;
  VectorXd Lambda_curr = Map<VectorXd>(mcmc->Lambda.data(), mcmc->dim_data*mcmc->dim_fact); // column-major
 //std::cout<<"before gradient"<<std::endl;
  stan::math::gradient(lambda_tar_fun, Lambda_curr, ln_px_curr, grad_ln_px_curr);
  //std::cout<<"after gradient"<<std::endl;
  // Proposal according MALA
  VectorXd prop_lambda_vec = Lambda_curr + mala_p_lambda*grad_ln_px_curr +
                    std::sqrt(2*mala_p_lambda)*Map<VectorXd>(normal_rng(std::vector<double>(mcmc->dim_data*mcmc->dim_fact, 0.),
                  std::vector<double>(mcmc->dim_data*mcmc->dim_fact,1.), Rng::Instance().get()).data() , mcmc->dim_data*mcmc->dim_fact);


  double ln_px_prop;
  VectorXd grad_ln_px_prop;
  stan::math::gradient(lambda_tar_fun, prop_lambda_vec, ln_px_prop, grad_ln_px_prop);

  MatrixXd prop_lambda = Map<MatrixXd>(prop_lambda_vec.data(), mcmc->dim_data, mcmc->dim_fact);

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
    mcmc->Lambda = prop_lambda;
    mcmc->pp_mix->update_decomposition_from_proposal();
    //std::cout<<"accepted Lambda"<<std::endl;
  }

  return;
}
*/
    

}