#ifndef MULTI_FACTOR_DPP_HPP
#define MULTI_FACTOR_DPP_HPP

#include "base_determinantalPP.hpp"

class MultiDpp: public BaseDeterminantalPP {

private:

  const MatrixXd * Lambda; // const pointer to the Lambda matrix

  double c_star_tmp; // c* of X (DPP) wrt the last passed Lambda

  // eigendecomposition params for last passed Lambda
  VectorXd phis_tmp;
  VectorXd phi_tildes_tmp;
  double Ds_tmp;

public:

  MultiDpp(const MatrixXd &ranges, int N, double c, double s);

  // set the pointer to Lambda and performs the initial decomposition
  void set_decomposition(const MatrixXd * lambda) override;

  // modifies the passed Ds, phis, phi_tildes, c_star according to the dpp defined with lambda
  void compute_eigen_and_cstar(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda);

  // it takes the proposed Lambda and performs decomposition, storing it in "tmp" variables
  void decompose_proposal(const MatrixXd& lambda) override;

  void update_decomposition_from_proposal() override;

  void compute_Kappas() override; // compute just once the grid for summation over Z^dim

  double dens_cond_in_proposal(const MatrixXd& x, bool log=true) override;
};

#endif
